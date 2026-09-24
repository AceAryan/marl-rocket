"""
DQN Agent with optional communication head.
Supports both action selection and message generation.
"""

import numpy as np
import random
from collections import deque
import json
import os


# ---------------------------------------------------------------------------
# Tiny neural net in pure numpy (no PyTorch dependency for portability)
# ---------------------------------------------------------------------------

def relu(x):
    return np.maximum(0, x)


class MLP:
    """2-hidden-layer MLP."""

    def __init__(self, in_dim, hidden, out_dim, seed=0):
        rng = np.random.default_rng(seed)
        scale = lambda i, o: np.sqrt(2.0 / i)
        self.W1 = rng.normal(0, scale(in_dim, hidden), (hidden, in_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, scale(hidden, hidden), (hidden, hidden)).astype(np.float32)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.W3 = rng.normal(0, scale(hidden, out_dim), (out_dim, hidden)).astype(np.float32)
        self.b3 = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x):
        h1 = relu(self.W1 @ x + self.b1)
        h2 = relu(self.W2 @ h1 + self.b2)
        return self.W3 @ h2 + self.b3

    def copy_from(self, other):
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def grads_zero(self):
        return [np.zeros_like(p) for p in self.params()]


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return list(zip(*batch))

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    """
    Independent DQN agent with:
    - epsilon-greedy exploration
    - target network (hard update every C steps)
    - optional message head (separate Q-values for message tokens)
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_messages: int = 5,
        hidden: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 10_000,
        batch_size: int = 64,
        target_update: int = 200,
        buffer_size: int = 50_000,
        agent_id: int = 0,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_messages = n_messages
        self.gamma = gamma
        self.lr = lr
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.agent_id = agent_id

        self.steps = 0

        # Action network + target
        self.q_net = MLP(obs_dim, hidden, n_actions, seed=agent_id)
        self.q_target = MLP(obs_dim, hidden, n_actions, seed=agent_id + 100)
        self.q_target.copy_from(self.q_net)

        # Message network (same architecture, different output head)
        self.msg_net = MLP(obs_dim, hidden // 2, n_messages, seed=agent_id + 200)

        self.buffer = ReplayBuffer(buffer_size)

        # Simple SGD accumulators (we use a manual update since no autograd)
        self._loss_history = []

    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray) -> int:
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        q = self.q_net.forward(obs)
        return int(np.argmax(q))

    def select_message(self, obs: np.ndarray, budget_exhausted: bool = False) -> int:
        if budget_exhausted:
            return 0  # silence
        if random.random() < self.eps:
            return random.randrange(self.n_messages)
        q = self.msg_net.forward(obs)
        return int(np.argmax(q))

    def decay_epsilon(self):
        self.eps = max(
            self.eps_end,
            self.eps - (1.0 - self.eps_end) / self.eps_decay
        )

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.push(obs, action, reward, next_obs, done)

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        self.steps += 1

        obs_b, act_b, rew_b, nobs_b, done_b = self.buffer.sample(self.batch_size)
        obs_b   = np.array(obs_b,   dtype=np.float32)
        act_b   = np.array(act_b,   dtype=np.int32)
        rew_b   = np.array(rew_b,   dtype=np.float32)
        nobs_b  = np.array(nobs_b,  dtype=np.float32)
        done_b  = np.array(done_b,  dtype=np.float32)

        # Compute targets
        targets = rew_b.copy()
        for i in range(self.batch_size):
            if not done_b[i]:
                targets[i] += self.gamma * np.max(self.q_target.forward(nobs_b[i]))

        # Manual SGD update for Q-net (simple finite-diff / direct update trick)
        loss = self._sgd_step(self.q_net, obs_b, act_b, targets)
        self._loss_history.append(loss)

        # Target network hard update
        if self.steps % self.target_update == 0:
            self.q_target.copy_from(self.q_net)

        self.decay_epsilon()
        return loss

    def _sgd_step(self, net, obs_b, act_b, targets):
        """
        Minimal SGD step using finite differences on the output layer only
        (a simplified approximation sufficient for course-level experiments).
        Replace with PyTorch for production use.
        """
        total_loss = 0.0
        lr = self.lr / self.batch_size

        for i in range(self.batch_size):
            q = net.forward(obs_b[i])
            a = act_b[i]
            td_err = q[a] - targets[i]
            total_loss += td_err ** 2

            # Backprop through last layer only (simplified)
            # dL/dq[a] = 2 * td_err
            grad_out = np.zeros_like(q)
            grad_out[a] = 2.0 * td_err

            # Last layer update: W3, b3
            h2 = relu(net.W2 @ relu(net.W1 @ obs_b[i] + net.b1) + net.b2)
            net.W3 -= lr * np.outer(grad_out, h2)
            net.b3 -= lr * grad_out

        return total_loss / self.batch_size

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        for name, net in [("q", self.q_net), ("msg", self.msg_net)]:
            np.savez(
                os.path.join(path, f"agent{self.agent_id}_{name}.npz"),
                W1=net.W1, b1=net.b1,
                W2=net.W2, b2=net.b2,
                W3=net.W3, b3=net.b3,
            )
        meta = {"eps": float(self.eps), "steps": int(self.steps), "loss_history": [float(x) for x in self._loss_history[-1000:]]}
        with open(os.path.join(path, f"agent{self.agent_id}_meta.json"), "w") as f:
            json.dump(meta, f)

    def load(self, path: str):
        for name, net in [("q", self.q_net), ("msg", self.msg_net)]:
            d = np.load(os.path.join(path, f"agent{self.agent_id}_{name}.npz"))
            net.W1, net.b1 = d["W1"], d["b1"]
            net.W2, net.b2 = d["W2"], d["b2"]
            net.W3, net.b3 = d["W3"], d["b3"]
        self.q_target.copy_from(self.q_net)
