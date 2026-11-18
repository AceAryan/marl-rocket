"""DQN agents (independent or parameter-sharing) with a learned message gate.

The Q-networks are small numpy MLPs with hand-written backprop and Adam:
for networks this small on CPU, per-op framework overhead dominates and numpy
is several times faster per update than PyTorch. Each network keeps its
parameters in one flat float32 vector (the per-layer arrays are views into
it), so Adam, gradient clipping and Polyak averaging are single vector ops.

Message gate. The gate is not a DQN head: it is a small regression network
that scores how much sending a message now is worth to the team, and the
agent sends when the score exceeds the per-message cost. Its training targets
come from the training loop (experiments/train.py):
  mode "advantage": A(o) = G_send - G_silent, the difference in team return
      between two copies of the game in which the message is delivered or
      withheld (counterfactual rollouts, centralised training);
  mode "q": Q(o, a) for a in {silent, send}, regressed on the Monte-Carlo team
      return that followed the choice actually made.
Neither target bootstraps, so the gate's training does not depend on which
policy collected the data (no on-policy/off-policy mismatch).
"""

import json
import math
import os
import random

import numpy as np


class MLP:
    """in -> hidden -> hidden -> out, ReLU, PyTorch-style uniform init."""

    def __init__(self, in_dim, hidden, out_dim, rng=None):
        rng = rng or np.random
        self.shapes = [(in_dim, hidden), (hidden,), (hidden, hidden), (hidden,), (hidden, out_dim), (out_dim,)]
        sizes = [int(np.prod(s)) for s in self.shapes]
        self.flat = np.empty(sum(sizes), dtype=np.float32)
        self.grad = np.zeros_like(self.flat)
        self.params, self._grads = [], []
        offset = 0
        for shape, size in zip(self.shapes, sizes):
            self.params.append(self.flat[offset:offset + size].reshape(shape))
            self._grads.append(self.grad[offset:offset + size].reshape(shape))
            offset += size
        for k, (fan_in, fan_out) in enumerate(((in_dim, hidden), (hidden, hidden), (hidden, out_dim))):
            bound = 1.0 / np.sqrt(fan_in)
            self.params[2 * k][...] = rng.uniform(-bound, bound, (fan_in, fan_out))
            self.params[2 * k + 1][...] = rng.uniform(-bound, bound, fan_out)

    def forward(self, x, cache=False):
        W1, b1, W2, b2, W3, b3 = self.params
        h1 = np.maximum(x @ W1 + b1, 0.0)
        h2 = np.maximum(h1 @ W2 + b2, 0.0)
        out = h2 @ W3 + b3
        if cache:
            self._cache = (x, h1, h2)
        return out

    __call__ = forward

    def backward(self, dout):
        """Flat gradient w.r.t. the parameters, given dL/dout for the last cached forward."""
        _, _, W2, _, W3, _ = self.params
        x, h1, h2 = self._cache
        g = self._grads
        dh2 = (dout @ W3.T) * (h2 > 0)
        dh1 = (dh2 @ W2.T) * (h1 > 0)
        np.matmul(x.T, dh1, out=g[0])
        dh1.sum(0, out=g[1])
        np.matmul(h1.T, dh2, out=g[2])
        dh2.sum(0, out=g[3])
        np.matmul(h2.T, dout, out=g[4])
        dout.sum(0, out=g[5])
        return self.grad

    def copy_from(self, other):
        self.flat[:] = other.flat

    def state_dict(self):
        return {f"p{i}": p.copy() for i, p in enumerate(self.params)}

    def load_state_dict(self, state):
        for i, p in enumerate(self.params):
            p[...] = np.asarray(state[f"p{i}"], dtype=np.float32)


class Adam:
    """Adam on one flat parameter vector."""

    def __init__(self, n_params, lr, betas=(0.9, 0.999), eps=1e-8):
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.m = np.zeros(n_params, dtype=np.float32)
        self.v = np.zeros(n_params, dtype=np.float32)
        self._tmp = np.zeros(n_params, dtype=np.float32)
        self.t = 0

    def step(self, flat, grad):
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        step = self.lr * math.sqrt(c2) / c1
        m, v, tmp = self.m, self.v, self._tmp
        m *= self.b1
        np.multiply(grad, 1.0 - self.b1, out=tmp)
        m += tmp
        v *= self.b2
        np.multiply(grad, grad, out=tmp)
        tmp *= 1.0 - self.b2
        v += tmp
        # Moments of dead units decay into float32 subnormals, which are
        # ~10x slower on x86 (numpy has no flush-to-zero); flush them.
        m[np.abs(m) < 1e-30] = 0.0
        v[v < 1e-30] = 0.0
        np.sqrt(v, out=tmp)
        tmp += self.eps
        np.divide(m, tmp, out=tmp)
        tmp *= step
        flat -= tmp


def _clip_(grad, max_norm):
    norm = math.sqrt(float(np.dot(grad, grad)))
    if norm > max_norm:
        grad *= max_norm / (norm + 1e-6)


class ReplayBuffer:
    """Fixed-size ring buffer of (obs, action, reward, next_obs, done, discount).

    `discount` is gamma**n for n-step transitions (`done` handles termination)."""

    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.disc = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0

    def push(self, obs, action, reward, next_obs, done, discount):
        i = self.idx
        self.obs[i] = obs
        self.act[i] = action
        self.rew[i] = reward
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self.disc[i] = discount
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ix = np.random.randint(0, self.size, size=batch_size)
        return self.obs[ix], self.act[ix], self.rew[ix], self.next_obs[ix], self.done[ix], self.disc[ix]

    def __len__(self):
        return self.size


class NStepAccumulator:
    """Turns a stream of 1-step transitions into n-step transitions.

    The time limit counts as terminal: the observation includes the remaining
    time, so the clock is part of the state and nothing follows the last step
    (Pardo et al., 2018, "Time limits in reinforcement learning")."""

    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.queue = []

    def push(self, obs, action, reward, next_obs, terminated, truncated):
        """Returns the list of n-step transitions that became ready."""
        done = bool(terminated or truncated)
        self.queue.append((obs, action, reward))
        out = []
        if len(self.queue) == self.n:
            out.append(self._emit(next_obs, done))
        if done:
            while self.queue:
                out.append(self._emit(next_obs, done))
        return out

    def _emit(self, next_obs, done):
        ret = 0.0
        for k, (_, _, r) in enumerate(self.queue):
            ret += (self.gamma ** k) * r
        obs, action, _ = self.queue[0]
        discount = self.gamma ** len(self.queue)
        self.queue.pop(0)
        return (obs, action, ret, next_obs, done, discount)

    def reset(self):
        self.queue = []


class QLearner:
    """A Double-DQN Q-function with its target network, optimiser and replay.
    Owned by one agent, or shared by the whole team (parameter sharing)."""

    def __init__(self, in_dim, n_actions, hidden=128, lr=5e-4, batch_size=64, tau=0.005,
                 buffer_size=100_000, learning_starts=1_000, max_grad_norm=10.0):
        self.q_net = MLP(in_dim, hidden, n_actions)
        self.q_target = MLP(in_dim, hidden, n_actions)
        self.q_target.copy_from(self.q_net)
        self.opt = Adam(self.q_net.flat.size, lr)
        self.buffer = ReplayBuffer(buffer_size, in_dim)
        self.batch_size = batch_size
        self.tau = tau
        self.learning_starts = learning_starts
        self.max_grad_norm = max_grad_norm
        self.steps = 0
        self.loss_history = []

    def update(self):
        self.steps += 1
        if len(self.buffer) < max(self.batch_size, self.learning_starts):
            return None
        obs, act, rew, nobs, done, disc = self.buffer.sample(self.batch_size)
        rows = np.arange(len(obs))
        # Double DQN: select the next action with the online net, evaluate with the target
        next_actions = self.q_net(nobs).argmax(axis=1)
        next_q = self.q_target(nobs)[rows, next_actions]
        target = rew + disc * next_q * (1.0 - done)

        q = self.q_net.forward(obs, cache=True)
        delta = q[rows, act] - target
        dout = np.zeros_like(q)
        dout[rows, act] = np.clip(delta, -1.0, 1.0) / len(obs)   # smooth-L1 gradient
        grad = self.q_net.backward(dout)
        _clip_(grad, self.max_grad_norm)
        self.opt.step(self.q_net.flat, grad)

        tflat = self.q_target.flat
        tflat *= 1.0 - self.tau
        tflat += self.tau * self.q_net.flat
        loss = float(np.mean(np.where(np.abs(delta) < 1.0, 0.5 * delta ** 2, np.abs(delta) - 0.5)))
        if self.steps % 100 == 0:
            self.loss_history.append(loss)
        return loss


class GateLearner:
    """The learned message gate (see the module docstring)."""

    def __init__(self, in_dim, mode="advantage", hidden=64, lr=5e-4, capacity=20_000, batch_size=64,
                 learning_starts=256, max_grad_norm=10.0):
        if mode not in ("advantage", "q"):
            raise ValueError("gate mode must be 'advantage' or 'q'")
        self.mode = mode
        self.net = MLP(in_dim, hidden, 1 if mode == "advantage" else 2)
        self.opt = Adam(self.net.flat.size, lr)
        self.capacity = capacity
        self.obs = np.zeros((capacity, in_dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.target = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.size = 0
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.max_grad_norm = max_grad_norm
        self.steps = 0

    def score(self, x):
        """Value of sending over staying silent (one score per row of x)."""
        out = self.net(x)
        return out[..., 0] if self.mode == "advantage" else out[..., 1] - out[..., 0]

    def push(self, x, action, target):
        i = self.idx
        self.obs[i] = x
        self.act[i] = action
        self.target[i] = target
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def update(self):
        self.steps += 1
        if self.size < max(self.batch_size, self.learning_starts):
            return None
        ix = np.random.randint(0, self.size, size=self.batch_size)
        rows = np.arange(self.batch_size)
        col = np.zeros(self.batch_size, dtype=np.int64) if self.mode == "advantage" else self.act[ix]
        out = self.net.forward(self.obs[ix], cache=True)
        delta = out[rows, col] - self.target[ix]
        dout = np.zeros_like(out)
        dout[rows, col] = np.clip(delta, -1.0, 1.0) / self.batch_size
        grad = self.net.backward(dout)
        _clip_(grad, self.max_grad_norm)
        self.opt.step(self.net.flat, grad)
        return float(np.mean(np.abs(delta)))


class DQNAgent:
    """One agent: its own n-step queue and exploration rate, acting with a
    Q-learner and message gate that are either its own or shared by the team.
    With parameter sharing, a one-hot agent id is appended to the input so the
    shared networks can still tell the agents apart."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        agent_id: int = 0,
        n_agents: int = 2,
        learner: QLearner = None,
        gate: GateLearner = None,
        id_input: bool = False,
        gate_mode: str = "advantage",
        hidden: int = 128,
        lr: float = 5e-4,
        gamma: float = 0.99,
        n_step: int = 3,
        batch_size: int = 64,
        tau: float = 0.005,
        buffer_size: int = 100_000,
        learning_starts: int = 1_000,
        max_grad_norm: float = 10.0,
        gate_capacity: int = 20_000,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.agent_id = agent_id
        self.gamma = gamma
        self.id_vec = np.eye(n_agents, dtype=np.float32)[agent_id] if id_input else np.zeros(0, np.float32)
        in_dim = obs_dim + len(self.id_vec)
        if learner is None:
            learner = QLearner(in_dim, n_actions, hidden, lr, batch_size, tau, buffer_size,
                               learning_starts, max_grad_norm)
        if gate is None:
            gate = GateLearner(in_dim, gate_mode, max(1, hidden // 2), lr, gate_capacity)
        self.learner = learner
        self.gate = gate
        self.nstep = NStepAccumulator(n_step, gamma)
        self.eps = 1.0

    # Compatibility accessors
    @property
    def q_net(self):
        return self.learner.q_net

    @property
    def msg_net(self):
        return self.gate.net

    def x(self, obs):
        """Network input for this agent."""
        if len(self.id_vec) == 0:
            return obs
        return np.concatenate([obs, self.id_vec]).astype(np.float32)

    def q_values(self, obs):
        return self.learner.q_net(self.x(obs))

    def select_action(self, obs) -> int:
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q_values(obs)))

    def gate_score(self, obs) -> float:
        return float(self.gate.score(self.x(obs)))

    def select_message(self, obs, cost: float = 0.0) -> int:
        """1 = send, 0 = stay silent. Exploration is silence-biased (half the
        exploratory picks are silent) so a budget is not burned at random."""
        if random.random() < self.eps:
            return random.randrange(2)
        return 1 if self.gate_score(obs) > cost else 0

    def store(self, obs, action, reward, next_obs, terminated, truncated=False):
        for t in self.nstep.push(obs, action, reward, next_obs, terminated, truncated):
            o, a, ret, no, done, disc = t
            self.learner.buffer.push(self.x(o), a, ret, self.x(no), done, disc)

    def end_episode(self):
        self.nstep.reset()

    def update(self):
        return self.learner.update()

    def update_gate(self):
        return self.gate.update()

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        for name, net in (("q", self.q_net), ("msg", self.msg_net)):
            np.savez(os.path.join(path, f"agent{self.agent_id}_{name}.npz"), **net.state_dict())
        meta = {
            "eps": float(self.eps),
            "steps": int(self.learner.steps),
            "gate_steps": int(self.gate.steps),
            "gate_mode": self.gate.mode,
            "id_input": bool(len(self.id_vec)),
            "loss_history": self.learner.loss_history[-1000:],
        }
        with open(os.path.join(path, f"agent{self.agent_id}_meta.json"), "w") as f:
            json.dump(meta, f)

    def load(self, path: str):
        for name, net in (("q", self.q_net), ("msg", self.msg_net)):
            with np.load(os.path.join(path, f"agent{self.agent_id}_{name}.npz")) as state:
                net.load_state_dict(state)
        self.learner.q_target.copy_from(self.learner.q_net)


def make_agents(obs_dim, n_actions, n_agents=2, share=True, gate_mode="advantage", **kw):
    """The team. share=True: one Q-learner and one gate for everyone (with an
    agent-id input); the shared learner samples n_agents x the batch from a
    n_agents x larger buffer, so each update sees the same data per agent."""
    if not share:
        return [DQNAgent(obs_dim, n_actions, agent_id=i, n_agents=n_agents, gate_mode=gate_mode, **kw)
                for i in range(n_agents)]
    kw = dict(kw)
    kw["batch_size"] = kw.get("batch_size", 64) * n_agents
    kw["buffer_size"] = kw.get("buffer_size", 100_000) * n_agents
    first = DQNAgent(obs_dim, n_actions, agent_id=0, n_agents=n_agents, id_input=True, gate_mode=gate_mode, **kw)
    others = [DQNAgent(obs_dim, n_actions, agent_id=i, n_agents=n_agents, id_input=True,
                       learner=first.learner, gate=first.gate, gate_mode=gate_mode, **kw)
              for i in range(1, n_agents)]
    return [first] + others


def unique(objs):
    """Distinct objects, in order (the learners or gates of a team)."""
    seen, out = set(), []
    for o in objs:
        if id(o) not in seen:
            seen.add(id(o))
            out.append(o)
    return out
