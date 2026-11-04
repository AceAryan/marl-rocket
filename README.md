# MARL Rocket — Learning Communication-Efficient Coordination

> *How does limiting communication affect the ability of cooperative RL agents to learn effective coordination?*

A 2D multi-agent environment inspired by Rocket League, built for systematic experiments on communication constraints in cooperative MARL.

---

## Project structure

```
marl-rocket/
├── env/
│   └── rocket_env.py        # Gymnasium-style 2D environment
├── agents/
│   └── dqn_agent.py         # PyTorch independent DQN + message head
├── experiments/
│   ├── train.py             # Single-setting training loop
│   └── run_all.py           # Budget sweeps at full and partial visibility
├── utils/
│   └── plot_results.py      # Matplotlib curves from saved history
├── tests/
│   └── test_env.py          # Smoke tests
└── requirements.txt
```

---

## Quick start

```bash
pip install -r requirements.txt

# Unit / smoke tests
python tests/test_env.py

# Train one setting (A, D5, D10, D, D50, C, B)
python -m experiments.train --setting B --obs_frac 0.15

# Run a sweep in parallel (resumable; finished runs are skipped)
python -m experiments.run_all --sweep budget --workers 12   # RQ1
python -m experiments.run_all --sweep all --workers 12      # RQ1-RQ3

# Re-summarise / re-plot an existing sweep
python utils/plot_results.py --dir checkpoints/sweep_assist --save
```

---

## Environment

20 x 30 field; the team attacks the goal at y = 30 (width 8). Spawn positions
are randomised. Ball physics are dribble-able (kick speed 1.1, friction 0.90,
about 11 units of travel per touch), so outcomes depend on control rather than
ricochets.

**Assist-based scoring** (`assist_required=True`, the default): a goal only
counts if the teammate touched the ball within the last 40 steps (and the
opponent, if present, has not touched it since); an unassisted shot bounces
off the goal line. Without this rule one agent can find the ball and score
alone, so a teammate's information is nearly worthless (see the calibration
below).

**Opponent**: off by default (`--opponent` in `train.py` enables a scripted
keeper confined to y >= 21 that clears the ball toward the team goal). With
assist-based scoring *and* the keeper, independent DQN learned nothing in
4000 episodes (0-5% goals, with or without an assist curriculum); without the
keeper, assisted scoring is learned reliably (80-96% goals by episode 2000).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_agents` | 2 | Cooperative team size (1 or 2) |
| `obs_frac` | 1.0 | Observation radius rho as a fraction of the field diagonal |
| `comm_budget` | None | Max messages per agent per episode (None = unlimited) |
| `comm_delay` | 0 | A message sent at step t is observed at step t + delay |
| `comm_every` | 1 | Messages can only be sent every k steps |
| `max_steps` | 200 | Episode length cap |
| `assist_required` | True | Only assisted goals count |
| `opponent` | True (env) / False (`train.py`) | Scripted keeper |

**Observation** (33 floats, `OBS_SLICES` in `env/rocket_env.py`): own position;
teammate, ball (+ velocity and position relative to self) and opponent, each with a
visibility flag and masked to zero when farther than rho x diagonal; the last
received message (token one-hot, the sender's ball sighting and position, and its age;
a message expires after 10 steps, so a spent budget cannot leave a stale
sighting in the observation for the rest of the episode);
remaining budget fraction; remaining time fraction; assist flags (my teammate
touched recently, so my goal would count; I touched recently, so my teammate's
would; whether this episode requires assists).

**Actions**: noop, up, down, left, right, boost (fast up).

**Messages**: none, attack, defend, pass, rotate. A sent message also carries
the sender's current view of the ball and its own position ("I'm here, the
ball is there"), so communication has real informational value under partial
observability. The message head is learned (SMDP
Q-learning between decision points), so agents learn *when* to spend budget.

**Reward** (shared): +100 goal, -100 concede, +0.1 per step while a teammate is
within 2 units of the ball, plus two potential-based shaping terms:

- Phi(s') - Phi(s) with Phi = -dist(ball, goal) - 0.5 * mean dist(teammate,
  ball). Undiscounted on purpose: with Phi < 0, the discounted form
  gamma * Phi(s') - Phi(s) pays (1 - gamma)|Phi| every step, i.e. it rewards
  idling far from the ball.
- gamma * Phi_a(s') - Phi_a(s) with Phi_a = 20 while a live assist exists (a goal
  by the last toucher would count). This pays out on the pass itself, which is
  otherwise invisible until a goal. It is discount-consistent on purpose: with
  the undiscounted form, creating an assist and letting it expire nets
  +20(1 - gamma^40) per cycle, and agents learned to farm passes in midfield.

Training uses reward x 0.1; all reported returns are unshaped task reward.

**Scripted calibration** (no keeper, 300 episodes): memoryless hand-coded teams
that act only on their own observations (pass toward the teammate's known
position, shoot when assist-eligible, otherwise go to the ball's known position
or search). Goal rate by message budget, spreading B messages evenly:

| rho | B=0 | B=5 | B=20 | B=50 | unlimited |
|-----|-----|-----|------|------|-----------|
| 1.0 | 0.96 | 0.96 | 0.96 | 0.96 | 0.96 |
| 0.25 | 0.90 | 0.98 | 0.98 | 0.98 | 0.98 |
| 0.15 | 0.43 | 0.59 | 0.92 | 0.95 | 0.91 |
| 0.10 | 0.16 | 0.23 | 0.55 | 0.67 | 0.67 |

Without assist-based scoring, sharing sightings changed scripted goal rate by
at most 5 points at any rho, which is why the earlier (no-assist) sweep showed
no benefit from communication.

**RL check** (rho = 0.15, no keeper, 4000 episodes, 2 seeds): B reaches 0.94 /
0.99 final goal rate (episode length 79 / 53) against A's 0.76 / 0.94 (101 /
63). B starts slower, since it must also learn when to message, but converges
higher.

---

## Communication settings

| Setting | Budget B | Period k | Description |
|---------|----------|----------|-------------|
| **A** | 0 | - | No communication |
| **D5 / D10** | 5 / 10 | 1 | Tight budgets (saturation curve) |
| **D** | 20 | 1 | Fixed budget (proposal setting D) |
| **D50** | 50 | 1 | Loose budget |
| **C** | inf | 5 | Unlimited, every 5 steps |
| **B** | inf | 1 | Unlimited, every step |

## Sweeps (`experiments/run_all.py --sweep ...`)

| Sweep | Runs | Question |
|-------|------|----------|
| `budget` | all 7 settings x rho in {1.0, 0.15} | RQ1: saturation point |
| `delay` | A, and B with delay in {0, 1, 2, 5}, rho = 0.15 | RQ2: delay |
| `obs` | A, B x rho in {1.0, 0.25, 0.15, 0.10} | RQ3: observability |
| `smoke` | A, B at rho 1.0 / 0.15 and D at 0.15 | quick pipeline check |
| `all` | union of budget + delay + obs (21 configurations) | |

Results go to `checkpoints/sweep_assist/`. Default: 5 seeds (0-4), 4000
episodes per run (channel settings learn more slowly than A at first). Agents are independent Double
DQN (numpy MLPs, 128-128 hidden; faster than PyTorch for networks this small on
CPU) with 3-step returns, Polyak target updates, one gradient step every 2 env
steps; epsilon decays linearly to 0.05 over the first 10% of episodes. Every periodic greedy evaluation uses the
same 50 fixed episode seeds; after training each run gets a 200-episode final
evaluation on held-out seeds, plus the same evaluation with the channel
silenced (the return drop measures reliance on communication). `summary.csv`
reports per-configuration means with 95% CIs, goal/concede rates, mean episode
length (time to goal), messages per episode, role separation, silenced drop,
learning-curve AUC (mean periodic greedy return, i.e. sample efficiency), and
Welch t-tests of final return and AUC against A. Final returns can saturate
near +100 once a setting learns reliably, so AUC and episode length are the
metrics with headroom.
