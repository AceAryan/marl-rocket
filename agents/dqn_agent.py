"""Independent DQN agent with separate action and communication heads."""

import json
import os
import random
from collections import deque

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buf = deque(maxlen=capacity)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, batch_size, replacement=False):
        if replacement:
            batch = random.choices(self.buf, k=batch_size)
        else:
            batch = random.sample(self.buf, batch_size)
        return list(zip(*batch))

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    """Independent DQN with target networks and separate message replay."""

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
        goal_buffer_min: int = 32,  # was 8 — too few unique goal transitions
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_messages = n_messages
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.agent_id = agent_id
        self.goal_buffer_min = goal_buffer_min
        self.device = torch.device("cpu")
        self.steps = 0
        self.msg_steps = 0

        self.q_net = QNetwork(obs_dim, hidden, n_actions).to(self.device)
        self.q_target = QNetwork(obs_dim, hidden, n_actions).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_target.eval()
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        msg_hidden = max(1, hidden // 2)
        self.msg_net = QNetwork(obs_dim, msg_hidden, n_messages).to(self.device)
        self.msg_target = QNetwork(obs_dim, msg_hidden, n_messages).to(self.device)
        self.msg_target.load_state_dict(self.msg_net.state_dict())
        self.msg_target.eval()
        self.msg_optimizer = torch.optim.Adam(self.msg_net.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_size)
        self.goal_buffer = ReplayBuffer(5_000)
        self.msg_buffer = ReplayBuffer(buffer_size)
        self._loss_history = []

    def _q_values(self, net, obs):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            return net(obs_tensor).cpu().numpy()

    def select_action(self, obs: np.ndarray) -> int:
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        return int(np.argmax(self._q_values(self.q_net, obs)))

    def select_message(self, obs: np.ndarray, budget_exhausted: bool = False) -> int:
        if budget_exhausted:
            return 0
        if random.random() < self.eps:
            return random.randrange(self.n_messages)
        return int(np.argmax(self._q_values(self.msg_net, obs)))

    def decay_epsilon(self):
        self.eps = max(
            self.eps_end,
            self.eps - (1.0 - self.eps_end) / self.eps_decay,
        )

    def store(self, obs, action, reward, next_obs, done, goal_transition=False):
        self.buffer.push(obs, action, reward, next_obs, done)
        if goal_transition:
            self.goal_buffer.push(obs, action, reward, next_obs, done)

    def store_message(self, obs, message, reward, next_obs, done):
        self.msg_buffer.push(obs, message, reward, next_obs, done)

    def _update_network(
        self,
        buffer,
        network,
        target_network,
        optimizer,
        step,
        goal_buffer=None,
    ):
        if len(buffer) < self.batch_size:
            return None

        if goal_buffer is not None and len(goal_buffer) >= self.goal_buffer_min:
            n_goal = self.batch_size // 4
            n_main = self.batch_size - n_goal
            main_batch = buffer.sample(n_main)
            goal_batch = goal_buffer.sample(n_goal, replacement=True)
            batch = [
                list(main_values) + list(goal_values)
                for main_values, goal_values in zip(main_batch, goal_batch)
            ]
            obs_b, act_b, rew_b, nobs_b, done_b = batch
        else:
            obs_b, act_b, rew_b, nobs_b, done_b = buffer.sample(self.batch_size)
        obs_t = torch.as_tensor(np.asarray(obs_b), dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(act_b, dtype=torch.int64, device=self.device)
        rew_t = torch.as_tensor(rew_b, dtype=torch.float32, device=self.device)
        nobs_t = torch.as_tensor(np.asarray(nobs_b), dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(done_b, dtype=torch.float32, device=self.device)

        q_selected = network(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = target_network(nobs_t).max(dim=1).values
            target = rew_t + self.gamma * next_q * (1.0 - done_t)

        loss = nn.functional.smooth_l1_loss(q_selected, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), max_norm=10.0)
        optimizer.step()

        if step % self.target_update == 0:
            target_network.load_state_dict(network.state_dict())
        return float(loss.item())

    def update(self):
        self.steps += 1
        loss = self._update_network(
            self.buffer,
            self.q_net,
            self.q_target,
            self.q_optimizer,
            self.steps,
            goal_buffer=self.goal_buffer,
        )
        if loss is not None:
            self._loss_history.append(loss)
            self.decay_epsilon()
        return loss

    def update_message_head(self):
        self.msg_steps += 1
        return self._update_network(
            self.msg_buffer,
            self.msg_net,
            self.msg_target,
            self.msg_optimizer,
            self.msg_steps,
        )

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        for name, net in (("q", self.q_net), ("msg", self.msg_net)):
            torch.save(net.state_dict(), os.path.join(path, f"agent{self.agent_id}_{name}.pt"))
        meta = {
            "eps": float(self.eps),
            "steps": int(self.steps),
            "msg_steps": int(self.msg_steps),
            "loss_history": self._loss_history[-1000:],
        }
        with open(os.path.join(path, f"agent{self.agent_id}_meta.json"), "w") as f:
            json.dump(meta, f)

    def load(self, path: str):
        for name, net in (("q", self.q_net), ("msg", self.msg_net)):
            torch_path = os.path.join(path, f"agent{self.agent_id}_{name}.pt")
            legacy_path = os.path.join(path, f"agent{self.agent_id}_{name}.npz")
            if os.path.exists(torch_path):
                state = torch.load(
                    torch_path,
                    map_location=self.device,
                    weights_only=True,
                )
                net.load_state_dict(state)
            elif os.path.exists(legacy_path):
                with np.load(legacy_path) as legacy:
                    layers = (net.net[0], net.net[2], net.net[4])
                    for layer, prefix in zip(layers, ("W1", "W2", "W3")):
                        layer.weight.data.copy_(
                            torch.as_tensor(legacy[prefix].T, dtype=torch.float32)
                        )
                        layer.bias.data.copy_(
                            torch.as_tensor(legacy["b" + prefix[1]], dtype=torch.float32)
                        )
            else:
                raise FileNotFoundError(
                    f"No {name} network checkpoint found in {path}"
                )
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.msg_target.load_state_dict(self.msg_net.state_dict())