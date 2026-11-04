"""
RocketEnv: A 2D multi-agent cooperative environment inspired by Rocket League.
Supports configurable communication budgets, partial observability, and team rewards.

Coordinates: field is [0, W] x [0, H]. The team attacks the goal at y = H and
defends the goal at y = 0. A scripted opponent defends y = H and tries to push
the ball toward y = 0.
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional


# --- Constants ---
FIELD_W, FIELD_H = 20.0, 30.0
FIELD_D = float(np.hypot(FIELD_W, FIELD_H))   # diagonal, used for observation radius
GOAL_W = 8.0
AGENT_RADIUS = 0.5
BALL_RADIUS = 0.4
CONTACT_DIST = AGENT_RADIUS + BALL_RADIUS
MAX_SPEED = 0.8          # agent displacement per step
BOOST_MULT = 1.5
KICK_SPEED = 1.1         # ball speed after a team kick (dribble-able: ~11 units of travel)
OPP_SPEED = 0.4          # opponent displacement per step
OPP_KICK_SPEED = 0.8
OPP_MIN_Y = 0.7 * FIELD_H  # opponent is a keeper confined to the last 30% of the field
BALL_FRICTION = 0.90
SUBSTEPS = 2             # physics substeps per env step (prevents tunnelling)

# Reward (task units follow the proposal: +/-100 for goals)
GOAL_REWARD = 100.0
PROX_REWARD = 0.1        # per step if any teammate is within PROX_DIST of the ball
PROX_DIST = 2.0
SHAPING_GAMMA = 1.0      # undiscounted: with Phi < 0, gamma < 1 would pay (1-gamma)|Phi| per step for idling
SHAPING_SCALE = 1.0      # potential = -SHAPING_SCALE * dist(ball, opponent goal)
APPROACH_SCALE = 0.5     #             - APPROACH_SCALE * dist(closest teammate, ball)
ASSIST_WINDOW = 40       # assisted goal: teammate touched the ball within this many steps
ASSIST_POTENTIAL = 20.0  # Phi_assist while a goal would count (pays out on the pass itself)
ASSIST_GAMMA = 0.99      # learner's discount: Phi_assist >= 0, so the discounted form is exact

# Actions: 0=noop, 1=up, 2=down, 3=left, 4=right, 5=boost (up, faster)
N_ACTIONS = 6
MOVE_DELTAS = np.array(
    [(0, 0), (0, 1), (0, -1), (-1, 0), (1, 0), (0, BOOST_MULT)], dtype=np.float32
)

# Message vocabulary: 0=none, 1=attack, 2=defend, 3=pass, 4=rotate
N_MESSAGES = 5
MSG_TTL = 10             # a received message expires after this many steps

# Observation layout (per agent)
OBS_SLICES = {
    "self": slice(0, 2),
    "mate": slice(2, 4),
    "mate_vis": slice(4, 5),
    "ball": slice(5, 7),
    "ball_vel": slice(7, 9),
    "ball_rel": slice(9, 11),
    "ball_vis": slice(11, 12),
    "opp": slice(12, 14),
    "opp_vis": slice(14, 15),
    "msg_token": slice(15, 15 + N_MESSAGES),
    "msg_ball": slice(20, 24),         # sender's ball pos + vel at send time
    "msg_ball_vis": slice(24, 25),     # did the sender see the ball
    "msg_age": slice(25, 26),
    "budget": slice(26, 27),
    "time": slice(27, 28),
    "msg_sender": slice(28, 30),       # sender's own position at send time
    "assist_me": slice(30, 31),        # teammate touched recently: my goal would count
    "assist_mate": slice(31, 32),      # I touched recently: teammate's goal would count
    "assist_rule": slice(32, 33),      # this episode requires assisted goals
}
OBS_DIM = 33


def _dist(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _norm_pos(p):
    """Map field coordinates to [-1, 1]."""
    return (2.0 * np.asarray(p, dtype=np.float32) / [FIELD_W, FIELD_H] - 1.0).astype(np.float32)


class RocketEnv(gym.Env):
    """
    2D multi-agent cooperative environment.

    Observation per agent (OBS_DIM floats, see OBS_SLICES):
        own position, teammate position + visible flag,
        ball position/velocity, ball position relative to self, ball visible flag,
        opponent position + visible flag,
        last received message: token one-hot, the sender's ball sighting
        (pos, vel, visible) and its age,
        remaining communication budget fraction, remaining time fraction.

    Under partial observability (obs_frac < 1) an object is masked to zeros
    (with its visible flag 0) when it is farther than obs_frac * field diagonal.

    Communication:
        comm_budget=None  -> unlimited
        comm_budget=0     -> no communication
        comm_budget=N     -> N messages per episode per agent
        comm_every=k      -> messages can only be sent on steps where t % k == 0
        comm_delay=d      -> a message sent at step t appears in the obs of step t+d
    A sent (non-empty) message carries its token plus the sender's current view
    of the ball, and is delivered to all teammates.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_agents: int = 2,
        obs_frac: float = 1.0,
        comm_budget: Optional[int] = None,
        comm_delay: int = 0,
        comm_every: int = 1,
        max_steps: int = 200,
        curriculum: str = "full",
        opponent: bool = True,
        assist_required: bool = True,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        if curriculum not in ("easy", "full"):
            raise ValueError("curriculum must be 'easy' or 'full'")
        if n_agents not in (1, 2):
            raise ValueError("n_agents must be 1 or 2")
        if comm_every < 1 or comm_delay < 0:
            raise ValueError("comm_every must be >= 1 and comm_delay >= 0")
        self.n_agents = n_agents
        self.obs_frac = obs_frac
        self.comm_budget = comm_budget
        self.comm_delay = comm_delay
        self.comm_every = comm_every
        self.max_steps = max_steps
        self.curriculum = curriculum
        self.opponent_enabled = opponent
        # Assist-based scoring: a goal only counts if the teammate touched the
        # ball within ASSIST_WINDOW steps (and the opponent has not touched it
        # since); an unassisted shot bounces off the goal line.
        self.assist_required = assist_required and n_agents > 1
        self.goal_width = 12.0 if curriculum == "easy" else GOAL_W
        self.opp_speed = 0.0 if curriculum == "easy" else OPP_SPEED
        self.render_mode = render_mode

        self._obs_size = OBS_DIM
        obs_low = np.full(OBS_DIM, -1.0, dtype=np.float32)
        obs_high = np.full(OBS_DIM, 1.0, dtype=np.float32)
        for key in ("ball_vel", "msg_ball"):
            obs_low[OBS_SLICES[key]] = -2.0
            obs_high[OBS_SLICES[key]] = 2.0
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.message_space = spaces.Discrete(N_MESSAGES)

        self.np_random, _ = gym.utils.seeding.np_random(None)
        self._reset_state()

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), self._get_info()

    def can_send(self, i: int) -> bool:
        """Whether agent i may send a non-empty message on the upcoming step."""
        if self.comm_budget is not None and self._msgs_sent[i] >= self.comm_budget:
            return False
        return (self._step_count + 1) % self.comm_every == 0

    def step(self, actions: list, messages: Optional[list] = None):
        """
        actions  : list of int, length n_agents
        messages : list of int (message tokens), length n_agents, or None
        """
        allowed = [self.can_send(i) for i in range(self.n_agents)]
        self._step_count += 1
        prev_potential = self._potential()
        prev_assist = self._assist_potential()

        moves = MOVE_DELTAS[np.asarray(actions, dtype=np.int64)] * MAX_SPEED
        opp_move = self._opponent_move()

        terminated = False
        scored = 0
        for _ in range(SUBSTEPS):
            self._agent_pos += moves / SUBSTEPS
            np.clip(
                self._agent_pos,
                [AGENT_RADIUS, AGENT_RADIUS],
                [FIELD_W - AGENT_RADIUS, FIELD_H - AGENT_RADIUS],
                out=self._agent_pos,
            )
            if self.opponent_enabled:
                self._opp_pos += opp_move / SUBSTEPS
                np.clip(
                    self._opp_pos,
                    [AGENT_RADIUS, OPP_MIN_Y],
                    [FIELD_W - AGENT_RADIUS, FIELD_H - AGENT_RADIUS],
                    out=self._opp_pos,
                )

            self._ball_pos += self._ball_vel / SUBSTEPS
            scored = self._check_goal()
            if scored > 0 and self.assist_required and not self._assisted(self._last_toucher):
                scored = 0   # unassisted: the goal line acts as a wall
            if scored:
                terminated = True
                break
            self._bounce_ball()
            self._collide()
        self._ball_vel *= BALL_FRICTION

        # --- Task reward (shared) ---
        task_reward = GOAL_REWARD * scored
        if scored > 0:
            self._score_team += 1
        elif scored < 0:
            self._score_opp += 1
        if not terminated and min(_dist(p, self._ball_pos) for p in self._agent_pos) < PROX_DIST:
            task_reward += PROX_REWARD

        # Potential-based shaping toward the opponent goal. Phi of the actual
        # final state is used on termination too, so conceding is never rewarded.
        shaping_reward = SHAPING_GAMMA * self._potential() - prev_potential
        # Discount-consistent (Ng et al. 1999) for the assist term: with the
        # undiscounted form, passing to create assists and letting them expire
        # nets +K(1 - gamma^window) per cycle, so agents farm passes in midfield.
        shaping_reward += ASSIST_GAMMA * self._assist_potential() - prev_assist

        # --- Communication ---
        if messages is not None:
            self._handle_messages(messages, allowed)
        self._deliver_messages()

        truncated = (not terminated) and self._step_count >= self.max_steps
        obs = self._get_obs()
        info = self._get_info()
        info["step"] = self._step_count
        info["task_reward"] = float(task_reward)
        info["shaping_reward"] = float(shaping_reward)
        return obs, float(task_reward + shaping_reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self):
        rng = self.np_random
        self._step_count = 0
        self._score_team = 0
        self._score_opp = 0

        # Agents spawn in their own half: one left, one right
        xs = [rng.uniform(2.0, FIELD_W / 2 - 1.0), rng.uniform(FIELD_W / 2 + 1.0, FIELD_W - 2.0)]
        self._agent_pos = np.array(
            [[xs[i], rng.uniform(2.0, 0.35 * FIELD_H)] for i in range(self.n_agents)],
            dtype=np.float32,
        )

        if self.curriculum == "easy":
            ball = [rng.uniform(5.0, FIELD_W - 5.0), rng.uniform(0.55, 0.7) * FIELD_H]
        else:
            ball = [rng.uniform(3.0, FIELD_W - 3.0), rng.uniform(0.4, 0.6) * FIELD_H]
        self._ball_pos = np.array(ball, dtype=np.float32)
        self._ball_vel = np.zeros(2, dtype=np.float32)

        if self.opponent_enabled:
            self._opp_pos = np.array(
                [rng.uniform(FIELD_W / 2 - 3.0, FIELD_W / 2 + 3.0), 0.85 * FIELD_H],
                dtype=np.float32,
            )
        else:
            self._opp_pos = np.array([-100.0, -100.0], dtype=np.float32)

        # Communication state
        self._msg_queue = []                               # (deliver_step, receiver, token, payload)
        self._msg_token = [0] * self.n_agents              # last delivered token
        self._msg_payload = [np.zeros(7, np.float32) for _ in range(self.n_agents)]
        # Touch history for assists
        self._last_touch = [-10**9] * self.n_agents     # step of each agent's last touch
        self._last_opp_touch = -10**9
        self._last_toucher = None
        self._msg_step = [None] * self.n_agents            # step of last delivery
        self._msgs_sent = [0] * self.n_agents

    def _opponent_move(self):
        """Scripted keeper confined to y >= OPP_MIN_Y: chase the ball from above
        (so its clearances head toward the team goal) when the ball nears its
        zone, otherwise guard its goal mouth."""
        if not self.opponent_enabled or self.opp_speed == 0.0:
            return np.zeros(2, dtype=np.float32)
        ball = self._ball_pos
        if ball[1] > OPP_MIN_Y - 2.0:
            target = ball + np.array([0.0, CONTACT_DIST * 0.8], dtype=np.float32)
            if self._opp_pos[1] < ball[1]:
                # Get around the ball instead of pushing it into its own goal
                side = 1.0 if self._opp_pos[0] >= ball[0] else -1.0
                target = ball + np.array([side * 1.5, 1.5], dtype=np.float32)
        else:
            goal_lo = (FIELD_W - self.goal_width) / 2
            goal_hi = (FIELD_W + self.goal_width) / 2
            target = np.array([np.clip(ball[0], goal_lo, goal_hi), FIELD_H - 3.0], dtype=np.float32)
        direction = target - self._opp_pos
        dist = math.hypot(float(direction[0]), float(direction[1]))
        if dist < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return (direction / dist * min(self.opp_speed, dist)).astype(np.float32)

    def _check_goal(self) -> int:
        """+1 if the team scored, -1 if the opponent scored, else 0."""
        goal_lo = (FIELD_W - self.goal_width) / 2
        goal_hi = (FIELD_W + self.goal_width) / 2
        bx, by = self._ball_pos
        if goal_lo <= bx <= goal_hi:
            if by >= FIELD_H - BALL_RADIUS:
                return 1
            if by <= BALL_RADIUS:
                return -1
        return 0

    def _bounce_ball(self):
        lo, hi = BALL_RADIUS, np.array([FIELD_W - BALL_RADIUS, FIELD_H - BALL_RADIUS])
        for axis in range(2):
            if self._ball_pos[axis] < lo:
                self._ball_pos[axis] = 2 * lo - self._ball_pos[axis]
                self._ball_vel[axis] = abs(self._ball_vel[axis])
            elif self._ball_pos[axis] > hi[axis]:
                self._ball_pos[axis] = 2 * hi[axis] - self._ball_pos[axis]
                self._ball_vel[axis] = -abs(self._ball_vel[axis])

    def _collide(self):
        """Kick the ball away from any player touching it."""
        players = [(p, KICK_SPEED, i) for i, p in enumerate(self._agent_pos)]
        if self.opponent_enabled:
            players.append((self._opp_pos, OPP_KICK_SPEED, None))
        for pos, speed, who in players:
            offset = self._ball_pos - pos
            d = math.hypot(float(offset[0]), float(offset[1]))
            if d < CONTACT_DIST:
                direction = offset / d if d > 1e-6 else np.array([0.0, 1.0], dtype=np.float32)
                self._ball_pos = (pos + direction * CONTACT_DIST).astype(np.float32)
                self._ball_vel = (direction * speed).astype(np.float32)
                if who is None:
                    self._last_opp_touch = self._step_count
                else:
                    self._last_touch[who] = self._step_count
                self._last_toucher = who
                np.clip(
                    self._ball_pos,
                    [BALL_RADIUS, BALL_RADIUS],
                    [FIELD_W - BALL_RADIUS, FIELD_H - BALL_RADIUS],
                    out=self._ball_pos,
                )

    def _touched_recently(self, j: int) -> bool:
        t = self._last_touch[j]
        return t > self._last_opp_touch and self._step_count - t <= ASSIST_WINDOW

    def _assisted(self, scorer) -> bool:
        """A goal by `scorer` counts if its teammate touched the ball recently."""
        if scorer is None:
            return False
        return any(self._touched_recently(j) for j in range(self.n_agents) if j != scorer)

    def _visible(self, i: int, pos) -> bool:
        if self.obs_frac >= 1.0:
            return True
        return _dist(pos, self._agent_pos[i]) <= self.obs_frac * FIELD_D

    def _ball_view(self, i: int):
        """Agent i's view of the ball: (pos, vel) normalised, and a visible flag."""
        if self._visible(i, self._ball_pos):
            return _norm_pos(self._ball_pos), self._ball_vel / KICK_SPEED, 1.0
        return np.zeros(2, np.float32), np.zeros(2, np.float32), 0.0

    def _handle_messages(self, messages: list, allowed: list):
        """Queue outgoing messages (token + sender's ball sighting)."""
        for i, token in enumerate(messages):
            if token == 0 or not allowed[i]:
                continue
            self._msgs_sent[i] += 1
            pos, vel, vis = self._ball_view(i)
            payload = np.concatenate([pos, vel, [vis], _norm_pos(self._agent_pos[i])]).astype(np.float32)
            for j in range(self.n_agents):
                if j != i:
                    self._msg_queue.append((self._step_count + self.comm_delay, j, int(token), payload))

    def _deliver_messages(self):
        pending = []
        for deliver_step, j, token, payload in self._msg_queue:
            if deliver_step <= self._step_count:
                self._msg_token[j] = token
                self._msg_payload[j] = payload
                self._msg_step[j] = deliver_step
            else:
                pending.append((deliver_step, j, token, payload))
        self._msg_queue = pending

    def _potential(self):
        """Phi(s): ball close to the opponent goal, and some teammate close to
        the ball (min over the team, so only one agent is pulled to the ball)."""
        ball_to_goal = _dist(self._ball_pos, (FIELD_W / 2, FIELD_H))
        dists = [_dist(p, self._ball_pos) for p in self._agent_pos]
        # With assists both agents must reach the ball, so use the mean distance
        approach = float(np.mean(dists)) if self.assist_required else min(dists)
        return -SHAPING_SCALE * ball_to_goal - APPROACH_SCALE * approach

    def _assist_potential(self):
        """K while a live assist exists (a goal by the last toucher would count)."""
        if self.assist_required and self._assisted(self._last_toucher):
            return ASSIST_POTENTIAL
        return 0.0

    def _get_obs(self):
        """Return list of per-agent observations (with partial obs mask)."""
        obs_list = []
        time_left = 1.0 - self._step_count / self.max_steps
        for i in range(self.n_agents):
            o = np.zeros(OBS_DIM, dtype=np.float32)
            o[OBS_SLICES["self"]] = _norm_pos(self._agent_pos[i])

            if self.n_agents > 1 and self._visible(i, self._agent_pos[1 - i]):
                o[OBS_SLICES["mate"]] = _norm_pos(self._agent_pos[1 - i])
                o[OBS_SLICES["mate_vis"]] = 1.0

            pos, vel, vis = self._ball_view(i)
            if vis:
                o[OBS_SLICES["ball"]] = pos
                o[OBS_SLICES["ball_vel"]] = vel
                rel = (self._ball_pos - self._agent_pos[i]) / [FIELD_W, FIELD_H]
                o[OBS_SLICES["ball_rel"]] = rel
                o[OBS_SLICES["ball_vis"]] = 1.0

            if self.opponent_enabled and self._visible(i, self._opp_pos):
                o[OBS_SLICES["opp"]] = _norm_pos(self._opp_pos)
                o[OBS_SLICES["opp_vis"]] = 1.0

            # Messages expire after MSG_TTL steps; otherwise a budgeted agent's
            # last sighting would persist, stale but flagged visible, all episode.
            age = None if self._msg_step[i] is None else self._step_count - self._msg_step[i]
            if age is not None and age <= MSG_TTL:
                o[OBS_SLICES["msg_token"]][self._msg_token[i]] = 1.0
                o[OBS_SLICES["msg_ball"]] = self._msg_payload[i][:4]
                o[OBS_SLICES["msg_ball_vis"]] = self._msg_payload[i][4]
                o[OBS_SLICES["msg_sender"]] = self._msg_payload[i][5:7]
                o[OBS_SLICES["msg_age"]] = age / MSG_TTL
            else:
                o[OBS_SLICES["msg_token"]][0] = 1.0
                o[OBS_SLICES["msg_age"]] = 1.0

            if self.comm_budget is None:
                o[OBS_SLICES["budget"]] = 1.0
            else:
                o[OBS_SLICES["budget"]] = (
                    (self.comm_budget - self._msgs_sent[i]) / max(self.comm_budget, 1)
                )
            o[OBS_SLICES["time"]] = time_left
            if self.assist_required:
                o[OBS_SLICES["assist_rule"]] = 1.0
                o[OBS_SLICES["assist_me"]] = float(self._assisted(i))
                o[OBS_SLICES["assist_mate"]] = float(self._touched_recently(i))
            obs_list.append(o)
        return obs_list

    def _get_info(self):
        return {
            "score_team": self._score_team,
            "score_opp": self._score_opp,
            "msgs_sent": self._msgs_sent.copy(),
            "can_send": [self.can_send(i) for i in range(self.n_agents)],
            "ball_pos": self._ball_pos.copy(),
            "agent_pos": self._agent_pos.copy(),
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
        for i, position in enumerate(self._agent_pos):
            r, c = to_grid(*position)
            grid[r][c] = str(i + 1)
        if self.opponent_enabled:
            r, c = to_grid(*self._opp_pos)
            grid[r][c] = 'X'

        print(f"\nStep {self._step_count} | Score {self._score_team}-{self._score_opp}")
        print("+" + "-" * cols + "+")
        for row in grid:
            print("|" + "".join(row) + "|")
        print("+" + "-" * cols + "+")
