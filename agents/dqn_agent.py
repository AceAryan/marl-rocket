"""Independent DQN agent with separate action and communication heads.

The Q-networks are small numpy MLPs with hand-written backprop and Adam:
for 28 -> 128 -> 128 -> 6 networks on CPU, per-op framework overhead dominates,
and numpy is several times faster per update than PyTorch here.
"""

import json
import os
import random

import numpy as np


class MLP:
    """in -> hidden -> hidden -> out, ReLU, PyTorch-style uniform init."""

    def __init__(self, in_dim, hidden, out_dim, rng=None):
        rng = rng or np.random
        self.params = []
        for fan_in, fan_out in ((in_dim, hidden), (hidden, hidden), (hidden, out_dim)):
            bound = 1.0 / np.sqrt(fan_in)
            self.params.append(rng.uniform(-bound, bound, (fan_in, fan_out)).astype(np.float32))
            self.params.append(rng.uniform(-bound, bound, fan_out).astype(np.float32))

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
        """Gradients w.r.t. params given dL/dout for the last cached forward."""
        W1, _, W2, _, W3, _ = self.params
        x, h1, h2 = self._cache
        dh2 = (dout @ W3.T) * (h2 > 0)
        dh1 = (dh2 @ W2.T) * (h1 > 0)
        return [x.T @ dh1, dh1.sum(0), h1.T @ dh2, dh2.sum(0), h2.T @ dout, dout.sum(0)]

    def copy_from(self, other):
        self.params = [p.copy() for p in other.params]

    def state_dict(self):
        return {f"p{i}": p for i, p in enumerate(self.params)}

    def load_state_dict(self, state):
        self.params = [np.asarray(state[f"p{i}"], dtype=np.float32) for i in range(len(self.params))]


class Adam:
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8):
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        step = self.lr * np.sqrt(c2) / c1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * (g * g)
            # Moments of dead units decay into float32 subnormals, which are
            # ~10x slower on x86 (numpy has no flush-to-zero); flush them.
            m[np.abs(m) < 1e-30] = 0.0
            v[v < 1e-30] = 0.0
            p -= step * m / (np.sqrt(v) + self.eps)


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
    """Turns a stream of 1-step transitions into n-step transitions."""

    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.queue = []

    def push(self, obs, action, reward, next_obs, terminated, truncated):
        """Returns the list of n-step transitions that became ready."""
        self.queue.append((obs, action, reward))
        out = []
        if len(self.queue) == self.n:
            out.append(self._emit(next_obs, terminated))
        if terminated or truncated:
            while self.queue:
                out.append(self._emit(next_obs, terminated))
        return out

    def _emit(self, next_obs, terminated):
        ret = 0.0
        for k, (_, _, r) in enumerate(self.queue):
            ret += (self.gamma ** k) * r
        obs, action, _ = self.queue[0]
        discount = self.gamma ** len(self.queue)
        self.queue.pop(0)
        return (obs, action, ret, next_obs, terminated, discount)

    def reset(self):
        self.queue = []


class DQNAgent:
    """Independent Double DQN with n-step returns, Polyak-averaged target
    networks, and a separate message Q-head with its own replay."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_messages: int = 5,
        hidden: int = 128,
        lr: float = 5e-4,
        gamma: float = 0.99,
        n_step: int = 3,
        batch_size: int = 64,
        tau: float = 0.005,
        buffer_size: int = 100_000,
        learning_starts: int = 1_000,
        max_grad_norm: float = 10.0,
        agent_id: int = 0,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_messages = n_messages
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.learning_starts = learning_starts
        self.max_grad_norm = max_grad_norm
        self.agent_id = agent_id
        self.eps = 1.0
        self.steps = 0
        self.msg_steps = 0

        self.q_net = MLP(obs_dim, hidden, n_actions)
        self.q_target = MLP(obs_dim, hidden, n_actions)
        self.q_target.copy_from(self.q_net)
        self.q_optimizer = Adam(self.q_net.params, lr)

        msg_hidden = max(1, hidden // 2)
        self.msg_net = MLP(obs_dim, msg_hidden, n_messages)
        self.msg_target = MLP(obs_dim, msg_hidden, n_messages)
        self.msg_target.copy_from(self.msg_net)
        self.msg_optimizer = Adam(self.msg_net.params, lr)

        self.buffer = ReplayBuffer(buffer_size, obs_dim)
        self.msg_buffer = ReplayBuffer(buffer_size, obs_dim)
        self.nstep = NStepAccumulator(n_step, gamma)
        self._pending_msg = None   # [obs, message, discounted return, discount]
        self._loss_history = []

    def select_action(self, obs: np.ndarray) -> int:
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q_net(obs)))

    def select_message(self, obs: np.ndarray, budget_exhausted: bool = False) -> int:
        if budget_exhausted:
            return 0
        if random.random() < self.eps:
            return random.randrange(self.n_messages)
        return int(np.argmax(self.msg_net(obs)))

    def store(self, obs, action, reward, next_obs, terminated, truncated=False):
        for t in self.nstep.push(obs, action, reward, next_obs, terminated, truncated):
            self.buffer.push(*t)

    def message_decision(self, obs, message):
        """Record a message choice made at a decision point. Message decisions
        are sparse (period k, finite budget), so each one is treated as an
        option whose return runs until the next decision (SMDP Q-learning)."""
        if self._pending_msg is not None:
            p_obs, p_msg, p_ret, p_disc = self._pending_msg
            self.msg_buffer.push(p_obs, p_msg, p_ret, obs, False, p_disc)
        self._pending_msg = [obs, message, 0.0, 1.0]

    def message_reward(self, reward):
        if self._pending_msg is not None:
            self._pending_msg[2] += self._pending_msg[3] * reward
            self._pending_msg[3] *= self.gamma

    def end_episode(self, final_obs):
        """Flush the open message option; the episode end is terminal for it."""
        if self._pending_msg is not None:
            p_obs, p_msg, p_ret, p_disc = self._pending_msg
            self.msg_buffer.push(p_obs, p_msg, p_ret, final_obs, True, p_disc)
        self._pending_msg = None
        self.nstep.reset()

    def _update_network(self, buffer, network, target_network, optimizer):
        if len(buffer) < max(self.batch_size, self.learning_starts):
            return None
        obs, act, rew, nobs, done, disc = buffer.sample(self.batch_size)
        rows = np.arange(len(obs))

        # Double DQN: select the next action with the online net, evaluate with the target
        next_actions = network(nobs).argmax(axis=1)
        next_q = target_network(nobs)[rows, next_actions]
        target = rew + disc * next_q * (1.0 - done)

        q = network.forward(obs, cache=True)
        delta = q[rows, act] - target
        loss = float(np.mean(np.where(np.abs(delta) < 1.0, 0.5 * delta ** 2, np.abs(delta) - 0.5)))

        dout = np.zeros_like(q)
        dout[rows, act] = np.clip(delta, -1.0, 1.0) / len(obs)   # smooth-L1 gradient
        grads = network.backward(dout)
        norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads))
        if norm > self.max_grad_norm:
            grads = [g * (self.max_grad_norm / (norm + 1e-6)) for g in grads]
        optimizer.step(network.params, grads)

        for tp, p in zip(target_network.params, network.params):
            tp *= 1.0 - self.tau
            tp += self.tau * p
        return loss

    def update(self):
        self.steps += 1
        loss = self._update_network(self.buffer, self.q_net, self.q_target, self.q_optimizer)
        if loss is not None and self.steps % 100 == 0:
            self._loss_history.append(loss)
        return loss

    def update_message_head(self):
        self.msg_steps += 1
        return self._update_network(
            self.msg_buffer, self.msg_net, self.msg_target, self.msg_optimizer
        )

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        for name, net in (("q", self.q_net), ("msg", self.msg_net)):
            np.savez(os.path.join(path, f"agent{self.agent_id}_{name}.npz"), **net.state_dict())
        meta = {
            "eps": float(self.eps),
            "steps": int(self.steps),
            "msg_steps": int(self.msg_steps),
            "loss_history": self._loss_history[-1000:],
        }
        with open(os.path.join(path, f"agent{self.agent_id}_meta.json"), "w") as f:
            json.dump(meta, f)

    def load(self, path: str):
        for name, net, target in (("q", self.q_net, self.q_target), ("msg", self.msg_net, self.msg_target)):
            with np.load(os.path.join(path, f"agent{self.agent_id}_{name}.npz")) as state:
                net.load_state_dict(state)
            target.copy_from(net)
