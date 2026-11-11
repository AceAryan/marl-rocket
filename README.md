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
│   └── dqn_agent.py         # numpy independent DQN + message head
├── experiments/
│   ├── train.py             # Single-setting training loop
│   └── run_all.py           # Budget sweeps at full and partial visibility
├── utils/
│   ├── plot_results.py      # summary.csv + combined results.png
│   └── report.py            # per-RQ figures + results.md tables
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

# Train one setting (A, D2..D, C, B learned; R2..Rinf rule; F2..Finf fixed)
python -m experiments.train --setting B --obs_frac 0.15

# Run a sweep in parallel (resumable; finished runs are skipped)
python -m experiments.run_all --sweep budget --workers 12   # RQ1
python -m experiments.run_all --sweep all --workers 12      # RQ1-RQ3

# Re-summarise / re-plot an existing sweep
python utils/plot_results.py --dir checkpoints/sweep_v4 --save
python -m utils.report --dir checkpoints/sweep_v4   # figures/ + results.md
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
| `max_steps` | 80 | Episode length cap (`MAX_STEPS`) |
| `assist_required` | True | Only assisted goals count |
| `opponent` | True (env) / False (`train.py`) | Scripted keeper |

**Observation** (42 floats, `OBS_SLICES` in `env/rocket_env.py`): own position;
teammate, ball (+ velocity and position relative to self) and opponent, each with a
visibility flag and masked to zero when farther than rho x diagonal; the last
received message (token one-hot, the sender's ball sighting and position, and its age;
a message expires after 10 steps, so a spent budget cannot leave a stale
sighting in the observation for the rest of the episode);
remaining budget fraction; remaining time fraction; assist flags (my teammate
touched recently, so my goal would count; I touched recently, so my teammate's
would; whether this episode requires assists); steps since my last send (capped at the
10-step message lifetime); and a memory: the last known ball position (absolute and
relative to me) and last known teammate position, each with its age (capped at 40
steps), from my own sightings or received messages, whichever is newer. The memory
only summarises what the agent has already observed; without it, a message vanished
from a memoryless policy's input after 10 steps, so sparse or stale messages
(budgets, schedules of every 8-16 steps) did worse than no communication.
`memory=False` (`--no_memory`) blanks it for an ablation.

**Actions**: noop, up, down, left, right, boost (fast up).

**Messages**: the environment's vocabulary is none, attack, defend, pass, rotate.
A sent message carries the sender's current view of the ball and its own
position ("I'm here, the ball is there"), so communication has real
informational value under partial observability. The learned message head
chooses only silent or send (token 1): the tokens carried no extra content, and
letting the head pick among four interchangeable tokens fed the receiver an
arbitrary, drifting one-hot input that slowed its learning.

**Learned sender (IQL-Comm)**: a per-agent message Q-head, trained with SMDP
Q-learning between decision points, decides *when* to spend the budget. Two
changes made it learn reliably:

- *Value-of-information credit* (`msg_credit="voi"`): each step the teammate
  holds my message, I am credited with how much it improves the teammate's
  decision, Q_j(o, a*) - Q_j(o, a_cf), where a_cf is the action the teammate
  would take with the message masked out (its own Q-network, i.e. centralised
  training), minus a cost of 0.05 per message. This replaces the noisy, delayed
  team reward (`msg_credit="team"`, the original credit). The sender also
  observes the steps since its last send, so it can tell when the teammate's
  copy has expired.
- *Rule-to-learned handover* (`msg_handover=0.5`): during the first half of
  training, a send-on-sight rule makes each decision with probability p, which
  falls linearly from 1 to 0; the head learns off-policy from every decision.
  Without it, sender and receiver co-adapt: the receiver learns against a send
  pattern that keeps shifting, and runs were unstable (some seeds collapsed to
  under 10% goals). Evaluation always uses the head alone.

Learned communication is **event-triggered**: an agent makes a message decision
only at steps where it may send and currently sees the ball; otherwise it stays
silent. Message exploration is silence-biased (an exploratory pick is silent
half the time). Without these, greedy D5 agents sent 65% of their budget in the
first 10 steps and about half their messages without seeing the ball (empty
sightings), so budgeted settings did worse than no communication
(`checkpoints/sweep_assist`, kept as the "before" result).

**Episodes are 80 steps** (`MAX_STEPS`), a match clock: with 200 steps nearly
every team eventually scored, which hid the speed difference that information
buys. At 80 steps, a scripted check of no-communication policies scored 94% with
full view but 78% at rho = 0.15 and 66% at rho = 0.10 (at 200 steps: 99 / 94 / 73).

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

Budgets are messages per agent per 80-step episode.

| Setting | Sender | Budget | Description |
|---------|--------|--------|-------------|
| **A** | - | 0 | No communication |
| **D2 / D5 / D10 / D** | learned | 2 / 5 / 10 / 20 | Learned sender under a budget (D = proposal setting D) |
| **C** | learned | inf, every 5 steps | Periodic (proposal setting C) |
| **B** | learned | inf | Unlimited (proposal setting B) |
| **R2 / R5 / R10 / R20 / Rinf** | rule | 2 / 5 / 10 / 20 / inf | Send whenever allowed and seeing the ball, until the budget is spent |
| **F2 / F5 / F10 / F20 / Finf** | fixed | every 40 / 16 / 8 / 4 / 1 steps | Broadcast every k steps (80/k messages), sighting or not |

The three kinds of sender separate the questions: F* measure the value of the
information itself, R* add a sensible hand-coded trigger, and the learned
senders must also discover *when* to send.

## Sweeps (`experiments/run_all.py --sweep ...`)

| Sweep | Runs | Question |
|-------|------|----------|
| `budget` | A, D2..D, C, B at rho = 0.075 | RQ1: saturation point, learned sender |
| `rule` | R2..Rinf at rho = 0.075 | RQ1, rule sender |
| `fixed` | F2..Finf at rho = 0.075 | RQ1, fixed schedule |
| `delay` | A; B, Rinf, Finf with delay in {0, 2, 5}, rho = 0.075 | RQ2: delay |
| `obs` | A, B, Rinf, Finf x rho in {1.0, 0.15, 0.10, 0.075} | RQ3: observability |
| `pilot` | A, B, Rinf, Finf at rho = 0.075 | headroom check at the main radius |
| `smoke` | A, D, B, R5, F5 at rho = 0.075 | quick pipeline check |
| `all` | union of the above (35 configurations, 350 runs at 10 seeds) | |

Results go to `checkpoints/sweep_v4/` (`summary.csv`, `results.png`, `results.md`, `figures/`). Earlier sweeps (`sweep_final`, `sweep_v2`, `sweep_v3`: no memory, shorter training) are incompatible with the current code; don't resume into them. Default: 10 seeds (0-9), 10,000
episodes per run (channel settings learn more slowly than A at first). Agents are independent Double
DQN (numpy MLPs, 128-128 hidden; faster than PyTorch for networks this small on
CPU) with 3-step returns, Polyak target updates, one gradient step every 2 env
steps; epsilon decays linearly to 0.05 over the first 10% of episodes. Every periodic greedy evaluation uses the
same 100 fixed validation episodes, and the snapshot with the best validation goal rate is kept; after training it gets a 200-episode final
evaluation on held-out seeds, plus the same evaluation with the channel
silenced (the return drop measures reliance on communication). `summary.csv`
reports per-configuration means with 95% CIs, goal/concede rates, mean episode
length (time to goal), messages per episode, role separation, silenced drop,
learning-curve AUC (mean periodic greedy return, i.e. sample efficiency), and
Welch t-tests of final return and AUC against A. Final returns can saturate
near +100 once a setting learns reliably, so AUC and episode length are the
metrics with headroom.

## Deviations from the proposal

- **Assist-based scoring**: a goal counts only if the teammate touched the ball
  within the last 40 steps. Without it, the shared sightings barely changed
  scripted goal rates (at most 5 points), so no experiment could detect a benefit
  from communication.
- **No scripted keeper** in the main experiments (`--opponent` adds it). With
  assists plus the keeper, independent DQN learned nothing in 4000 episodes
  (0-5% goal rate), so concede rate and goal difference are not meaningful.
- **Event-triggered learned messages** with a silent/send head,
  value-of-information credit and a rule-to-learned handover (see above), plus
  **rule (R*) and fixed-schedule (F*) baselines**, which separate the value of
  the shared information from the cost of learning when to send.
- **80-step episodes** instead of 200, so speed (and so information) matters,
  and budgets rescaled to 2-20 messages per episode.
- The main observation radius is rho = 0.075 (proposal: 0.25; `MAIN_RHO` in
  `run_all.py`). With memory and 80-step episodes, no communication scores
  92-96% at rho >= 0.10, which leaves communication no room to help; at 0.075 it
  falls to about 72% while unlimited communication reaches about 87% (2-seed
  pilot). At 0.05 no setting learned reliably in 10k episodes.
- Results are reported as medians with interquartile ranges and Mann-Whitney U
  tests: about 4% of runs collapse when one agent's Q-function diverges, and a
  single collapsed seed would dominate a mean or a t-test.
