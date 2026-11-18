"""
RocketEnv: A 2D multi-agent cooperative environment inspired by Rocket League.
Supports configurable communication budgets, partial observability, and team rewards.

Coordinates: field is [0, W] x [0, H]. The team attacks the goal at y = H and
defends the goal at y = 0. A scripted opponent defends y = H and tries to push
the ball toward y = 0.
"""

import copy
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
BUDGET_CAP = 64          # budget_left = log1p(messages left) / log1p(BUDGET_CAP), capped at 1

# Message payload: what the sender knew at the step it decided to send.
PAYLOAD = {
    "ball": slice(0, 2),        # ball position (normalised), 0 if not seen
    "ball_vel": slice(2, 4),    # ball velocity / KICK_SPEED
    "ball_vis": slice(4, 5),    # did the sender see the ball
    "sender": slice(5, 7),      # sender's own position (normalised)
    "touch": slice(7, 8),       # steps since the sender last touched the ball / (ASSIST_WINDOW+1), 1 = none
}
PAYLOAD_DIM = 8

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
    "msg_ball": slice(20, 24),         # sender's ball pos + vel at its decision step
    "msg_ball_vis": slice(24, 25),     # did the sender see the ball
    "msg_age": slice(25, 26),
    "budget": slice(26, 27),           # fraction of my budget left (1 = unlimited)
    "time": slice(27, 28),
    "msg_sender": slice(28, 30),       # sender's own position at its decision step
    "assist_me": slice(30, 31),        # a teammate touch I know of makes my goal count next step
    "assist_mate": slice(31, 32),      # my own touch makes my teammate's goal count next step
    "assist_rule": slice(32, 33),      # this episode requires assisted goals
    "send_age": slice(33, 34),         # steps since my last send / MSG_TTL, capped at 1: whether
                                       # the teammate's copy of my last message has expired
}
# Memory: the last known ball and teammate positions with their age, built
# from the agent's own sightings and the messages it received (whichever is
# newer). A summary of the agent's own observation history, so no extra
# information; without it a memoryless policy loses a message after MSG_TTL
# steps, and sparse or stale messages confused it more than they informed it.
OBS_SLICES.update({
    "mem_ball": slice(34, 36),         # last known ball position
    "mem_ball_rel": slice(36, 38),     # ... relative to my current position
    "mem_ball_age": slice(38, 39),     # steps since that sighting / MEM_AGE_CAP (1 = unknown)
    "mem_mate": slice(39, 41),         # last known teammate position
    "mem_mate_age": slice(41, 42),     # steps since that sighting / MEM_AGE_CAP (1 = unknown)
    "msg_touch": slice(42, 43),        # the held message's "sender touched the ball" age (1 = no touch)
    "budget_left": slice(43, 44),      # messages I have left, log1p(left)/log1p(BUDGET_CAP) (1 = unlimited)
    "mate_budget_left": slice(44, 45), # teammate's budget left, inferred from the messages I received
})
BUDGET_KEYS = ("budget", "budget_left", "mate_budget_left")
OBS_DIM = 45
MEM_AGE_CAP = 40
MAX_STEPS = 80           # short, match-clock episodes: speed (and so information) matters
NO_TOUCH = -10**9


def _dist(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _denorm_pos(p):
    """Inverse of _norm_pos."""
    return (np.asarray(p, dtype=np.float32) + 1.0) / 2.0 * np.array([FIELD_W, FIELD_H], dtype=np.float32)


def _norm_pos(p):
    """Map field coordinates to [-1, 1]."""
    return (2.0 * np.asarray(p, dtype=np.float32) / [FIELD_W, FIELD_H] - 1.0).astype(np.float32)


def _budget_feature(left):
    return min(1.0, math.log1p(max(left, 0)) / math.log1p(BUDGET_CAP))


class RocketEnv(gym.Env):
    """
    2D multi-agent cooperative environment.

    Observation per agent (OBS_DIM floats, see OBS_SLICES):
        own position, teammate position + visible flag,
        ball position/velocity, ball position relative to self, ball visible flag,
        opponent position + visible flag,
        last received message: token one-hot, the sender's ball sighting
        (pos, vel, visible), position and last-touch age, and the message's age,
        own and teammate's remaining budget, remaining time, assist flags, memory.

    Under partial observability (obs_frac < 1) an object is masked to zeros
    (with its visible flag 0) when it is farther than obs_frac * field diagonal.

    Assist flags only use what the agent can know: its own touches, teammate
    touches it saw (the ball or the teammate was in view), and touches a
    received message reported. The true touch history is never observed.

    Communication:
        comm_budget=None  -> unlimited
        comm_budget=0     -> no communication
        comm_budget=N     -> N messages per episode per agent
        comm_every=k      -> messages can only be sent on steps ceil(k/2), ceil(k/2)+k, ...
        comm_delay=d      -> a message sent at step t appears in the obs of step t+d
    A sent (non-empty) message carries its token plus what the sender observed
    when it decided to send (its view of the ball, its position and last
    touch), and is delivered to all teammates. Sends that could not arrive
    before the last decision are refused.

    match=True: a team goal does not end the episode. Play restarts like a
    kickoff (players back in their own half, the ball dropped at a random spot
    in midfield, touches and memories cleared); the clock and the message
    budget carry on. info["goals"] counts goals in the episode.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_agents: int = 2,
        obs_frac: float = 1.0,
        comm_budget: Optional[int] = None,
        comm_delay: int = 0,
        comm_every: int = 1,
        max_steps: int = MAX_STEPS,
        curriculum: str = "full",
        opponent: bool = True,
        assist_required: bool = True,
        memory: bool = True,
        budget_obs: bool = True,
        match: bool = False,
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
        self.comm_phase = (comm_every + 1) // 2   # first allowed send step (1-based)
        self.max_steps = max_steps
        self.curriculum = curriculum
        self.opponent_enabled = opponent
        # Assist-based scoring: a goal only counts if the teammate touched the
        # ball within ASSIST_WINDOW steps (and the opponent has not touched it
        # since); an unassisted shot bounces off the goal line.
        self.assist_required = assist_required and n_agents > 1
        self.goal_width = 12.0 if curriculum == "easy" else GOAL_W
        self.opp_speed = 0.0 if curriculum == "easy" else OPP_SPEED
        self.memory = memory
        self.budget_obs = budget_obs
        self.match = match
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
        upcoming = self._step_count + 1
        # A message sent at step s is seen in the observation after step s + delay;
        # the last decision uses the observation after step max_steps - 1.
        if upcoming + self.comm_delay > self.max_steps - 1:
            return False
        return upcoming >= self.comm_phase and (upcoming - self.comm_phase) % self.comm_every == 0

    def clone(self):
        """An independent copy of the full environment state (including its RNG),
        for counterfactual rollouts during training."""
        new = object.__new__(type(self))
        shared = ("observation_space", "action_space", "message_space")
        memo = {id(self.__dict__[k]): self.__dict__[k] for k in shared if k in self.__dict__}
        new.__dict__ = copy.deepcopy(self.__dict__, memo)
        return new

    def step(self, actions: list, messages: Optional[list] = None):
        """
        actions  : list of int, length n_agents
        messages : list of int (message tokens), length n_agents, or None
        """
        allowed = [self.can_send(i) for i in range(self.n_agents)]
        # What each sender knows right now (its decision-time observation)
        outgoing = []
        if messages is not None:
            for i, token in enumerate(messages):
                if token != 0 and allowed[i]:
                    outgoing.append((i, int(token), self._payload(i), self._step_count, self._last_touch[i]))
        # Potentials of the state the step starts from. Computed before the clock
        # advances, so an assist that expires during this step is charged.
        prev_potential = self._potential()
        prev_assist = self._assist_potential()
        self._step_count += 1

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
            self._goal_steps.append(self._step_count)
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

        if scored and self.match:
            # Kickoff: play restarts; shaping was charged on the goal state above,
            # so each goal is worth the same as in single-goal episodes.
            self._kickoff()
            terminated = False

        # --- Communication ---
        self._observe_touches()
        for i, token, payload, t_obs, touch in outgoing:
            self._send(i, token, payload, t_obs, touch)
        self._deliver_messages()
        self._update_memory()

        truncated = (not terminated) and self._step_count >= self.max_steps
        obs = self._get_obs()
        info = self._get_info()
        info["step"] = self._step_count
        info["task_reward"] = float(task_reward)
        info["shaping_reward"] = float(shaping_reward)
        info["scored"] = int(scored)
        return obs, float(task_reward + shaping_reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spawn_players_and_ball(self):
        rng = self.np_random
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

    def _clear_play_state(self):
        """Touches, known touches, held messages and memories (reset and kickoff)."""
        n = self.n_agents
        self._last_touch = [NO_TOUCH] * n            # step of each agent's last touch
        self._last_opp_touch = NO_TOUCH
        self._last_toucher = None
        self._known_mate_touch = [NO_TOUCH] * n      # newest teammate touch each agent knows of
        self._msg_token = [0] * n                    # last delivered token
        self._msg_payload = [np.zeros(PAYLOAD_DIM, np.float32) for _ in range(n)]
        self._msg_step = [None] * n                  # step of last delivery
        self._last_send = [None] * n                 # step of each agent's last send
        # Memory (see OBS_SLICES): field position and step of the newest sighting
        self._mem_ball = [None] * n
        self._mem_ball_t = [None] * n
        self._mem_mate = [None] * n
        self._mem_mate_t = [None] * n

    def _reset_state(self):
        self._step_count = 0
        self._score_team = 0
        self._score_opp = 0
        self._goal_steps = []
        self._kickoff_step = 0                         # information from before this step is void
        self._msg_queue = []                           # (deliver_step, receiver, token, payload, t_obs, touch)
        self._msgs_sent = [0] * self.n_agents
        self._msgs_received = [0] * self.n_agents
        self._spawn_players_and_ball()
        self._clear_play_state()
        self._update_memory()

    def _kickoff(self):
        """Match mode: restart play after a goal (clock and budgets carry on)."""
        self._spawn_players_and_ball()
        self._clear_play_state()
        self._kickoff_step = self._step_count

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

    def _live(self, touch_step, step) -> bool:
        """A touch at touch_step still makes the teammate's goal count at `step`."""
        return touch_step > self._last_opp_touch and step - touch_step <= ASSIST_WINDOW

    def _touched_recently(self, j: int) -> bool:
        return self._live(self._last_touch[j], self._step_count)

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

    def _payload(self, i: int):
        """What agent i can report right now."""
        pos, vel, vis = self._ball_view(i)
        touch_age = self._step_count - self._last_touch[i]
        touch = min(touch_age, ASSIST_WINDOW + 1) / (ASSIST_WINDOW + 1)
        return np.concatenate([pos, vel, [vis], _norm_pos(self._agent_pos[i]), [touch]]).astype(np.float32)

    def _observe_touches(self):
        """A teammate's touch this step is known to agents that saw the ball or the toucher."""
        for j in range(self.n_agents):
            if self._last_touch[j] != self._step_count:
                continue
            for i in range(self.n_agents):
                if i != j and (self._visible(i, self._ball_pos) or self._visible(i, self._agent_pos[j])):
                    self._known_mate_touch[i] = max(self._known_mate_touch[i], self._step_count)

    def _send(self, i, token, payload, t_obs, touch):
        self._msgs_sent[i] += 1
        self._last_send[i] = self._step_count
        for j in range(self.n_agents):
            if j != i:
                self._msg_queue.append((self._step_count + self.comm_delay, j, token, payload, t_obs, touch))

    def _deliver_messages(self):
        pending = []
        for item in self._msg_queue:
            deliver_step, j, token, payload, t_obs, touch = item
            if deliver_step > self._step_count:
                pending.append(item)
                continue
            self._msgs_received[j] += 1
            if t_obs < self._kickoff_step:
                continue   # sent before a kickoff: the play it describes is over
            self._msg_token[j] = token
            self._msg_payload[j] = payload
            self._msg_step[j] = deliver_step
            if payload[PAYLOAD["ball_vis"]][0] > 0:   # the sender saw the ball
                self._remember(self._mem_ball, self._mem_ball_t, j, _denorm_pos(payload[PAYLOAD["ball"]]), t_obs)
            self._remember(self._mem_mate, self._mem_mate_t, j, _denorm_pos(payload[PAYLOAD["sender"]]), t_obs)
            if touch > NO_TOUCH:
                self._known_mate_touch[j] = max(self._known_mate_touch[j], touch)
        self._msg_queue = pending

    @staticmethod
    def _remember(mem, mem_t, i, pos, t):
        if mem_t[i] is None or t >= mem_t[i]:
            mem[i] = np.asarray(pos, dtype=np.float32).copy()
            mem_t[i] = t

    def _update_memory(self):
        """Own sightings at the current step (always the newest information)."""
        for i in range(self.n_agents):
            if self._visible(i, self._ball_pos):
                self._remember(self._mem_ball, self._mem_ball_t, i, self._ball_pos, self._step_count)
            if self.n_agents > 1 and self._visible(i, self._agent_pos[1 - i]):
                self._remember(self._mem_mate, self._mem_mate_t, i, self._agent_pos[1 - i], self._step_count)

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

    def _budget_features(self, i):
        """(fraction left, log-count left, teammate's inferred log-count left)."""
        if not self.budget_obs:
            return 0.0, 0.0, 0.0
        if self.comm_budget is None:
            return 1.0, 1.0, 1.0
        left = self.comm_budget - self._msgs_sent[i]
        mate_left = self.comm_budget - self._msgs_received[i]
        return left / max(self.comm_budget, 1), _budget_feature(left), _budget_feature(mate_left)

    def _get_obs(self):
        """Return list of per-agent observations (with partial obs mask)."""
        obs_list = []
        time_left = 1.0 - self._step_count / self.max_steps
        upcoming = self._step_count + 1
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
                p = self._msg_payload[i]
                o[OBS_SLICES["msg_token"]][self._msg_token[i]] = 1.0
                o[OBS_SLICES["msg_ball"]] = np.concatenate([p[PAYLOAD["ball"]], p[PAYLOAD["ball_vel"]]])
                o[OBS_SLICES["msg_ball_vis"]] = p[PAYLOAD["ball_vis"]]
                o[OBS_SLICES["msg_sender"]] = p[PAYLOAD["sender"]]
                o[OBS_SLICES["msg_touch"]] = p[PAYLOAD["touch"]]
                o[OBS_SLICES["msg_age"]] = age / MSG_TTL
            else:
                o[OBS_SLICES["msg_token"]][0] = 1.0
                o[OBS_SLICES["msg_age"]] = 1.0
                o[OBS_SLICES["msg_touch"]] = 1.0

            frac, left, mate_left = self._budget_features(i)
            o[OBS_SLICES["budget"]] = frac
            o[OBS_SLICES["budget_left"]] = left
            o[OBS_SLICES["mate_budget_left"]] = mate_left
            o[OBS_SLICES["time"]] = time_left
            if self._last_send[i] is None:
                o[OBS_SLICES["send_age"]] = 1.0
            else:
                o[OBS_SLICES["send_age"]] = min(self._step_count - self._last_send[i], MSG_TTL) / MSG_TTL
            o[OBS_SLICES["mem_ball_age"]] = 1.0
            o[OBS_SLICES["mem_mate_age"]] = 1.0
            if self.memory:
                if self._mem_ball_t[i] is not None:
                    o[OBS_SLICES["mem_ball"]] = _norm_pos(self._mem_ball[i])
                    o[OBS_SLICES["mem_ball_rel"]] = (self._mem_ball[i] - self._agent_pos[i]) / [FIELD_W, FIELD_H]
                    o[OBS_SLICES["mem_ball_age"]] = min(self._step_count - self._mem_ball_t[i], MEM_AGE_CAP) / MEM_AGE_CAP
                if self._mem_mate_t[i] is not None:
                    o[OBS_SLICES["mem_mate"]] = _norm_pos(self._mem_mate[i])
                    o[OBS_SLICES["mem_mate_age"]] = min(self._step_count - self._mem_mate_t[i], MEM_AGE_CAP) / MEM_AGE_CAP
            if self.assist_required:
                # Whether a goal on the coming step would count, as far as agent i knows
                o[OBS_SLICES["assist_rule"]] = 1.0
                o[OBS_SLICES["assist_me"]] = float(self._live(self._known_mate_touch[i], upcoming))
                o[OBS_SLICES["assist_mate"]] = float(self._live(self._last_touch[i], upcoming))
            obs_list.append(o)
        return obs_list

    def _get_info(self):
        return {
            "score_team": self._score_team,
            "score_opp": self._score_opp,
            "goals": self._score_team,
            "goal_steps": list(self._goal_steps),
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
