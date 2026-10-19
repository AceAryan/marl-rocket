"""
RocketEnv: A 2D multi-agent cooperative environment inspired by Rocket League.
Supports configurable communication budgets, partial observability, and team rewards.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional


# --- Constants ---
FIELD_W, FIELD_H = 20.0, 30.0
GOAL_W = 6.0
AGENT_RADIUS = 0.5
BALL_RADIUS = 0.4
MAX_SPEED = 0.8
BALL_FRICTION = 0.95
SHAPING_GAMMA = 0.99
SHAPING_SCALE = 0.01

# Actions: 0=noop, 1=up, 2=down, 3=left, 4=right, 5=boost
N_ACTIONS = 6
MOVE_DELTAS = {
    0: (0, 0),
    1: (0, 1),
    2: (0, -1),
    3: (-1, 0),
    4: (1, 0),
    5: (0, 1.5),  # boost
}

# Message vocabulary: 0=none, 1=attack, 2=defend, 3=pass, 4=rotate
N_MESSAGES = 5


class RocketEnv(gym.Env):
    """
    2D multi-agent cooperative environment.

    Observation per agent (flat array):
        [self_x, self_y, teammate_x, teammate_y,
         ball_x, ball_y, ball_vx, ball_vy,
         opp_x, opp_y,
         last_message_received (one-hot, N_MESSAGES),
         remaining communication budget fraction]

    Action: integer in [0, N_ACTIONS)

    Communication:
        comm_budget=None  → unlimited
        comm_budget=0     → no communication
        comm_budget=N     → N messages per episode per agent
        comm_delay=k      → message arrives k steps later
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        n_agents: int = 2,
        obs_frac: float = 1.0,
        comm_budget: Optional[int] = None,
        comm_delay: int = 0,
        comm_every: int = 1,
        max_steps: int = 500,
        curriculum: str = "full",
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        if curriculum not in ("easy", "full"):
            raise ValueError("curriculum must be 'easy' or 'full'")
        self.n_agents = n_agents
        self.obs_frac = obs_frac          # partial observability [0,1]
        self.comm_budget = comm_budget    # None=unlimited, 0=none, N=N msgs/ep
        self.comm_delay = comm_delay      # steps before msg arrives
        self.comm_every = comm_every      # send every K steps
        self.max_steps = max_steps
        self.curriculum = curriculum
        self.goal_width = 10.0 if curriculum == "easy" else GOAL_W
        self.render_mode = render_mode

        # Observation size per agent
        self._obs_size = 10 + N_MESSAGES + 1  # positions + ball + opp + message + budget

        obs_low = np.full(self._obs_size, -1.0, dtype=np.float32)
        obs_high = np.full(self._obs_size, 1.0, dtype=np.float32)
        obs_low[6:8] = -1.5
        obs_high[6:8] = 1.5
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)
        # Message action space (separate)
        self.message_space = spaces.Discrete(N_MESSAGES)

        self._reset_state()

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, actions: list, messages: Optional[list] = None):
        """
        actions  : list of int, length n_agents
        messages : list of int (message tokens), length n_agents, or None
        """
        self._step_count += 1
        prev_ball_pos = self._ball_pos.copy()

        # --- Move agents ---
        for i, a in enumerate(actions):
            dx, dy = MOVE_DELTAS[a]
            self._agent_pos[i] += np.array([dx, dy]) * MAX_SPEED
            self._agent_pos[i] = np.clip(
                self._agent_pos[i],
                [AGENT_RADIUS, AGENT_RADIUS],
                [FIELD_W - AGENT_RADIUS, FIELD_H - AGENT_RADIUS],
            )

        # --- Move opponent (simple heuristic) ---
        self._move_opponent()

        # --- Ball physics ---
        self._ball_pos += self._ball_vel
        self._ball_vel *= BALL_FRICTION
        self._bounce_ball()

        # --- Check collisions with agents ---
        for i in range(self.n_agents):
            d = np.linalg.norm(self._ball_pos - self._agent_pos[i])
            if d < AGENT_RADIUS + BALL_RADIUS:
                direction = self._ball_pos - self._agent_pos[i]
                direction /= np.linalg.norm(direction) + 1e-8
                self._ball_vel = direction * MAX_SPEED * 1.5
                self._last_touch = i

        # --- Check collision with opponent ---
        d_opp = np.linalg.norm(self._ball_pos - self._opp_pos)
        if d_opp < AGENT_RADIUS + BALL_RADIUS:
            direction = self._ball_pos - self._opp_pos
            direction /= np.linalg.norm(direction) + 1e-8
            self._ball_vel = direction * MAX_SPEED

        # --- Communication ---
        if messages is not None:
            self._handle_messages(messages)

        # --- Rewards & termination ---
        reward, terminated = self._compute_reward()
        truncated = self._step_count >= self.max_steps
        next_potential = 0.0 if terminated or truncated else self._potential(self._ball_pos)
        shaping_reward = SHAPING_SCALE * (
            SHAPING_GAMMA * next_potential - self._potential(prev_ball_pos)
        )
        task_reward = reward
        reward += shaping_reward

        obs = self._get_obs()
        info = self._get_info()
        info["step"] = self._step_count
        info["task_reward"] = task_reward
        info["shaping_reward"] = shaping_reward

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self):
        self._step_count = 0
        self._score_team = 0
        self._score_opp = 0
        self._last_touch = -1

        # Agent positions (one row per agent)
        self._agent_pos = np.array([
            [FIELD_W * 0.35, FIELD_H * 0.2],
            [FIELD_W * 0.65, FIELD_H * 0.2],
        ], dtype=np.float32)

        # Ball
        ball_y = FIELD_H * (0.75 if self.curriculum == "easy" else 0.5)
        self._ball_pos = np.array([FIELD_W / 2, ball_y], dtype=np.float32)
        angle = self.np_random.uniform(0, 2 * np.pi) if hasattr(self, 'np_random') else 0.0
        self._ball_vel = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * 0.3

        # Opponent
        self._opp_pos = np.array([FIELD_W / 2, FIELD_H * 0.85], dtype=np.float32)

        # Communication buffers: delayed message queues per agent
        self._msg_buffers = [[] for _ in range(self.n_agents)]  # [(delay_left, token), ...]
        self._msg_received = [0] * self.n_agents                # last delivered token
        self._msgs_sent = [0] * self.n_agents                   # budget tracking

    def _move_opponent(self):
        """Simple rule-based opponent: chase ball when it's in upper half."""
        if self.curriculum == "easy":
            return
        target = self._ball_pos.copy()
        direction = target - self._opp_pos
        dist = np.linalg.norm(direction)
        if dist > 0.1:
            self._opp_pos += (direction / dist) * MAX_SPEED * 0.6
        self._opp_pos = np.clip(
            self._opp_pos,
            [AGENT_RADIUS, FIELD_H * 0.6],
            [FIELD_W - AGENT_RADIUS, FIELD_H - AGENT_RADIUS],
        )

    def _bounce_ball(self):
        """Wall bounces and clamp."""
        if self._ball_pos[0] <= BALL_RADIUS or self._ball_pos[0] >= FIELD_W - BALL_RADIUS:
            self._ball_vel[0] *= -1
        self._ball_pos = np.clip(
            self._ball_pos, [BALL_RADIUS, BALL_RADIUS], [FIELD_W - BALL_RADIUS, FIELD_H - BALL_RADIUS]
        )

    def _handle_messages(self, messages: list):
        """Queue outgoing messages with delay; deliver ready ones."""
        for i, token in enumerate(messages):
            # Check budget
            can_send = (
                token != 0
                and self._step_count % self.comm_every == 0
                and (self.comm_budget is None or self._msgs_sent[i] < self.comm_budget)
            )
            if can_send:
                for j in range(self.n_agents):
                    if j != i:
                        self._msg_buffers[j].append([self.comm_delay, token])
                self._msgs_sent[i] += 1

        # Age buffers and deliver
        for j in range(self.n_agents):
            new_buf = []
            for entry in self._msg_buffers[j]:
                entry[0] -= 1
                if entry[0] <= 0:
                    self._msg_received[j] = entry[1]
                else:
                    new_buf.append(entry)
            self._msg_buffers[j] = new_buf

    def _compute_reward(self):
        """Shared team reward."""
        reward = 0.0
        terminated = False

        bx, by = self._ball_pos
        goal_lo = (FIELD_W - self.goal_width) / 2
        goal_hi = (FIELD_W + self.goal_width) / 2

        # Team scores (ball enters opponent goal at top)
        if by >= FIELD_H - BALL_RADIUS and goal_lo <= bx <= goal_hi:
            reward += 100.0
            self._score_team += 1
            terminated = True

        # Opponent scores (ball enters our goal at bottom)
        if by <= BALL_RADIUS and goal_lo <= bx <= goal_hi:
            reward -= 100.0
            self._score_opp += 1
            terminated = True

        # Possession: agent near ball
        for i in range(self.n_agents):
            d = np.linalg.norm(self._ball_pos - self._agent_pos[i])
            if d < 2.0:
                reward += 0.1

        return reward, terminated

    @staticmethod
    def _potential(ball_pos):
        opponent_goal = np.array([FIELD_W / 2, FIELD_H], dtype=np.float32)
        return -float(np.linalg.norm(ball_pos - opponent_goal))

    def _get_obs(self):
        """Return list of per-agent observations (with partial obs mask)."""
        obs_list = []
        for i in range(self.n_agents):
            raw = np.concatenate([
                self._agent_pos[i] / [FIELD_W, FIELD_H],
                self._agent_pos[1 - i] / [FIELD_W, FIELD_H],  # teammate
                self._ball_pos / [FIELD_W, FIELD_H],
                self._ball_vel / MAX_SPEED,
                self._opp_pos / [FIELD_W, FIELD_H],
            ]).astype(np.float32)

            # Partial observability: zero out far objects
            if self.obs_frac < 1.0:
                visible_objects = (
                    (slice(2, 4), None),  # teammate
                    (slice(4, 6), slice(6, 8)),  # ball and its velocity
                    (slice(8, 10), None),  # opponent
                )
                for pos_slice, related_slice in visible_objects:
                    obj_pos = raw[pos_slice] * [FIELD_W, FIELD_H]
                    dist = np.linalg.norm(obj_pos - self._agent_pos[i])
                    if dist > FIELD_H * self.obs_frac:
                        raw[pos_slice] = 0.0
                        if related_slice is not None:
                            raw[related_slice] = 0.0

            # One-hot message
            msg_onehot = np.zeros(N_MESSAGES, dtype=np.float32)
            msg_onehot[self._msg_received[i]] = 1.0

            if self.comm_budget is None:
                budget_remaining = np.array([1.0], dtype=np.float32)
            else:
                used = self._msgs_sent[i]
                budget_remaining = np.array(
                    [(self.comm_budget - used) / max(self.comm_budget, 1)],
                    dtype=np.float32,
                )

            obs_list.append(np.concatenate([raw, msg_onehot, budget_remaining]))
        return obs_list

    def _get_info(self):
        return {
            "score_team": self._score_team,
            "score_opp": self._score_opp,
            "msgs_sent": self._msgs_sent.copy(),
            "ball_pos": self._ball_pos.copy(),
        }

    def render(self):
        if self.render_mode == "human":
            self._render_ascii()

    def _render_ascii(self):
        rows, cols = 20, 40
        grid = [['.' for _ in range(cols)] for _ in range(rows)]

        def to_grid(x, y):
            c = int(x / FIELD_W * (cols - 1))
            r = rows - 1 - int(y / FIELD_H * (rows - 1))
            return max(0, min(r, rows-1)), max(0, min(c, cols-1))

        r, c = to_grid(*self._ball_pos); grid[r][c] = 'O'
        r, c = to_grid(*self._agent_pos[0]); grid[r][c] = '1'
        r, c = to_grid(*self._agent_pos[1]); grid[r][c] = '2'
        r, c = to_grid(*self._opp_pos); grid[r][c] = 'X'

        print(f"\nStep {self._step_count} | Score {self._score_team}-{self._score_opp}")
        print("+" + "-" * cols + "+")
        for row in grid:
            print("|" + "".join(row) + "|")
        print("+" + "-" * cols + "+")
