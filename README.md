# MARL Rocket — Learning Communication-Efficient Coordination

> *How does limiting communication affect the ability of cooperative RL agents to learn effective coordination?*

A 2D two-agent soccer environment inspired by Rocket League, built for controlled experiments on message
budgets in cooperative multi-agent RL: two learning agents with a small field of view share a team reward and
may send budgeted messages ("here is what I see") to each other.

---

## Project structure

```
marl-rocket/
├── env/rocket_env.py          # environment (partial view, memory, budgeted channel, match mode)
├── agents/dqn_agent.py        # numpy Double DQN, optional parameter sharing, learned message gate
├── experiments/
│   ├── train.py               # one training run (one setting, one seed)
│   ├── config.py              # every sweep: pilot, main tiers 1-4, smoke
│   ├── run_all.py             # parallel, resumable sweeps
│   └── scripted.py            # random and hand-coded reference teams
├── utils/
│   ├── plot_results.py        # loading, statistics, summary.csv
│   └── report.py              # figures + results.md; pilot decision rules
└── tests/                     # pytest suite
```

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests -q

# One run
python -m experiments.train --setting L4 --obs_frac 0.075

# 1. Pilot (resumable: re-run the same command after an interruption)
python -m experiments.run_all --sweep pilot --save_dir checkpoints/pilot_v6 --workers 20
# 2. Apply the pre-registered pilot rules -> writes experiments/pilot_decisions.json
python -m utils.report --dir checkpoints/pilot_v6 --pilot --apply
# 3. Main sweep, tier by tier (reads the decisions), then the report
python -m experiments.run_all --sweep tier1 --workers 20
python -m experiments.run_all --sweep tier2 --workers 20
python -m experiments.run_all --sweep tier3 --workers 20
python -m experiments.run_all --sweep tier4 --workers 20
python -m utils.report --dir checkpoints/sweep_v5                  # figures/ + results.md
```

`--workers` defaults to the number of CPUs minus 2; `--dry_run` lists what would be trained. A run is
resumed only if its saved configuration matches, so copying `checkpoints/pilot_v6` (or a partly finished
`checkpoints/sweep_v5`) to another machine and re-running the same command continues where it stopped.
The `history.json` files are in git (weights are not), so a fresh clone of this branch resumes the pilot
too. `run_all` keeps a Windows PC awake while it works and rebuilds its worker pool if a worker dies.

**Cost** (measured on a 22-thread PC with 20 workers): a single-goal run takes about 20-30 minutes when 20
run in parallel (about 6 alone), a learned-sender run with counterfactual credit about 40 minutes, a match
run about 25 minutes. The pilot is 72 runs (45 already done, 27 left: about 1 hour). The main sweep is
550 runs (580 if the pilot picks rho = 0.06): roughly 6-7 hours for tier 1 (270 runs) and about 13 hours
in all with 20 workers.

---

## Environment

20 x 30 field; the team attacks the goal at y = 30 (width 8). Agents spawn in their own half, the ball in
a random spot of the midfield band. Ball physics are dribble-able (kick speed 1.1, friction 0.9, about 11
units of travel per touch). Actions: noop, up, down, left, right, boost (fast up).

**Assist-based scoring**: a goal counts only if the scorer's teammate touched the ball within the last 40
steps; an unassisted shot bounces off the goal line. Without it one agent can score alone and its teammate's
information is nearly worthless. No opponent in the experiments (`--opponent` adds a scripted keeper; with
it, the task was not learnable).

**Partial view**: an agent sees the ball, its teammate (and the opponent) only within rho x the field
diagonal (rho = 0.075 is a radius of 2.7 units, far less than one kick).

**Observation** (45 floats, `OBS_SLICES`): own position; teammate, ball (position, velocity, position
relative to me) when in view; the last received message (sighting, sender position, sender's last touch,
age); **own messages left (fraction and log-count) and the teammate's messages left, inferred from the
messages received** (feedback 2; `budget_obs=False` hides them for an ablation); remaining time;
assist flags; steps since my last send; and a memory of the last known ball and teammate positions with
their age, from own sightings or messages, whichever is newer.

The assist flags use only what an agent can know: its own touches, teammate touches it saw, and touches a
message reported ("I touched it" is part of every message, so it costs budget). Before v5 every setting,
including no communication, saw the teammate's true touches through this flag (a free channel).

**Messages**: a message carries what the sender observed when it decided to send: its view of the ball,
its own position and its last touch. It is delivered after `comm_delay` steps, shown for 10 steps, and
written into the receiver's memory. Budgets count messages per agent per episode; sends that could not
arrive before the last decision are refused.

**Match mode** (`match=True`, 200 steps): a goal does not end the episode; play restarts like a kickoff
(agents back in their half, ball dropped at a random midfield spot, memories cleared) while the clock and
the message budgets carry on. The score is goals per match.

**Reward** (shared): +100 per goal, a small proximity bonus and two potential-based shaping terms (ball
toward the goal and agents toward the ball; +20 potential while a live assist exists). Training uses reward
x 0.1; reported metrics are goals, never shaped return.

## Agents and training

Double DQN with 3-step returns, numpy MLPs (128-128), Adam, Polyak targets, one gradient step every 2 env
steps, epsilon decaying to 0.05 over the first 10% of episodes. Each network keeps its parameters in one
flat vector, so a training run is about 5-8x faster than before v5. By default the two agents **share
parameters** (one Q-network with an agent-id input; batch and buffer doubled so each agent's data is
replayed as often as with separate networks), as in RIAL/DIAL, CommNet and IC3Net; execution stays
decentralised. The time limit is treated as terminal because the remaining time is observed.

### How the message choice is trained (feedback 3)

Agents do not choose message *content*: a message always carries the sender's current sighting. A learned
sender (`L<B>`) only decides **when** to spend its budget. It is event-triggered: it makes a decision only
at steps where it may send and currently sees the ball, and stays silent otherwise.

The decision is made by a **message gate**, a small network that scores how much sending now is worth to
the team; the agent sends when the score exceeds a per-message cost (0.05 in training-reward units). The
gate is *not* trained by DQN and does not learn from bootstrapped replay targets, so there is no
on-policy/off-policy contradiction:

- **Counterfactual credit** (`msg_credit="rollout"`, centralised training): at up to two randomly chosen
  decision points per agent per episode, the simulator is copied twice; one copy delivers the message and
  the other withholds it, both continue for 40 steps with the agents' current greedy policies (including
  later gate decisions and the same budgets), and the gate regresses the **difference in discounted team
  reward**. This is a difference reward that uses only the team reward and directly prices the budget a
  send uses up.
- **Team-return credit** (`msg_credit="team"`, the proposal's original idea): the gate regresses the
  Monte-Carlo team return that followed each send or silent decision.

The pilot picks between the two (rule below). Early in training a hand-coded rule makes the decisions with
probability falling from 1 to 0 over the first 30% of episodes, so receivers first learn against a sensible
sender; the gate learns from all decisions because its targets do not depend on who decided.
The action networks are ordinary off-policy DQN learners trained from replay buffers.

Before v5 the gate was a Q-head credited with the receiver's own Q-difference with and without the
message. That credit is never negative and, as a check that forked the game at real sends showed, had
about zero correlation with a message's actual effect on the outcome; it also left the message in the
receiver's memory when computing "without the message".

## Communication settings

Budgets are messages per agent per episode (per match in match mode).

| Setting | Sender | Description |
|---|---|---|
| **A** | - | No communication |
| **L\<B\>** | learned gate | Event-triggered; the gate decides when to spend the budget B (`Linf`: unlimited) |
| **R\<B\>** | rule | Send on sight when the teammate's copy of my last message has expired, until B is spent |
| **F\<B\>** | fixed | Broadcast every ceil(T/B) steps, sighting or not (B usable messages) |
| **B / C / D** | learned | Proposal settings: Linf; learned, every 5 steps; L20 |

The three senders separate the questions: F measures the value of the information itself, R adds a
sensible trigger, and L must also learn when to send.

## Experiments

### Pilot (pre-registered; seeds 100-102, disjoint from the main seeds)

Written before the pilot was run; `python -m utils.report --dir <pilot> --pilot` applies the rules.

- **P1 parameter sharing**: single-goal, rho 0.075, A and Finf, shared vs independent learners. Sharing is
  kept unless it is more than 2 pp worse in median score or learning-curve AUC for either.
- **P2 regime**: {single-goal 80 steps, match 200 steps} x rho {0.06, 0.075, 0.1} x {A, Finf}. Choose the
  regime with the largest relative gap (median Finf - median A) / median A among those where A reaches at
  least 25% of Finf and no Finf seed collapses (scores under half its configuration's median); keep
  single-goal unless match mode's gap is at least 1.5x larger.
- **P3 senders** at rho 0.075 (both modes): R4, Rinf, L4 with each credit, Linf (match mode: budget 10).
  Credit = the one with the higher median at L4 (L10).
- **Training length**: the smallest of {10k, 15k} episodes (match: {3.5k, 5k}) at which the median validation
  curves of A, Finf and L4 move less than 2 pp over the last 20% of training.
- **Stability**: if more than 10% of the pilot runs that learned something (selected score >= 20) end
  with a last policy under half of their selected snapshot, linear learning-rate decay (5e-4 to 1e-4) is
  switched on.
- **Budget grid**: fixed in advance: 1, 2, 3, 4, 6, 8, 12, 16, 24 and unlimited for the learned and rule
  senders; 2, 4, 8, 16, 32 and every step for the fixed schedule (match mode: x2.5).

Two clarifications were made after 45 of the 72 pilot runs had finished, before any decision was taken.
Neither depends on which setting scores better. (1) The budget grid was first to be derived from the
unlimited senders' usage. But the paced rule sends at most once per message lifetime (about 2 messages per
single-goal episode), so that rule would have left only budgets {1, 2, 3}, too few for the saturation
question; the grid above is now fixed. (2) The stability rule now ignores runs that never learned (match
mode at small rho stays near 0 goals): their "last below half of best" is noise, not a collapse.

### Main sweep (seeds 0-9; `experiments/config.py`)

| Tier | Configurations | Question |
|---|---|---|
| 1 | A; L and R over the budget grid; F at 4 budgets + Finf | RQ1: where do more messages stop helping? learned vs rule vs fixed |
| 2 | L at 3 budgets with the budget features hidden; delay {1,2,4,8} for Linf and Rinf | feedback 2; RQ2 |
| 3 | A, Linf, Rinf at rho {1.0, 0.25, 0.15, 0.1}; R at a second rho over 4 budgets | RQ3; does the saturation point move with rho |
| 4 | C (proposal); L4 with the other credit | continuity; credit ablation |

### Analysis (`utils/report.py`)

Each run keeps the snapshot with the best score on 100 fixed validation episodes; it is then scored on 200
separate held-out episodes, and so is the last policy (the gap shows instability that selection would
hide). Per configuration: median and interquartile range over seeds, mean with a 95% t-interval.
Mann-Whitney U tests are Holm-corrected within each table; budget and delay trends use Spearman's rank
correlation; the observability question uses bootstrap CIs of the communication gain at each rho and of
its slope against log(rho). The **saturation point** is B90, the budget that reaches 90% of the gain from
no communication to unlimited, from a fitted `y(B) = y_inf - (y_inf - y0) exp(-B / beta)` with a seed
bootstrap, reported with the smallest tested budget whose median reaches the same 90%. Reliance on message
*content* is the drop when messages arrive at the same times but carry another episode's content.
Reference lines: a random team and a hand-coded team with and without messages.

## Pilot outcome

*(Pending: 45 of 72 runs finished, in `checkpoints/pilot_v6`. Fill in from `checkpoints/pilot_v6/pilot.md`
once the pilot completes.)* Seen so far, with 2-3 seeds and so not yet decisive: removing the free
teammate-touch channel roughly doubled the room for communication at rho = 0.075 (A about 60% vs the
every-step sender about 85%); at rho = 0.06 A learns poorly (14-26%); match mode does not learn at
rho <= 0.075 within 5k matches.

## Results

*(filled in after the main sweep; see `checkpoints/sweep_v5/results.md` and `figures/`)*

## What changed in v5, and why the earlier results were weak

Found by a code review with adversarial verification of each finding against the saved runs:

1. **Free channel in the baseline**: the assist flag showed every agent the teammate's true touches, so
   the no-communication team got decision-relevant teammate information for free (clearing it changed the
   trained A policy's action in about 64% of the steps where it was set).
2. **Learned-sender credit did not measure usefulness** (see above), which is why learned senders at small
   budgets did worse than no communication.
3. **Time limit bootstrapped** although remaining time is observed (Q at the deadline was erratic).
4. **Selection hid instability**: 26 of 98 earlier runs ended training below 50% goals, but only 6 after
   best-snapshot selection; the last policy is now reported too.
5. **Statistics**: up to 16 uncorrected tests per table, and only tests against A, so neither saturation,
   delay nor the rho interaction was tested; "silenced" evaluation measured reactions to never-seen inputs.
6. Smaller fixes: an expiring assist's shaping was never charged back; messages carried the post-step
   view (12% of decided sends arrived without a sighting); fixed schedules wasted their last message;
   sweeps died with the worker pool and reused runs regardless of configuration.

## Deviations from the proposal

- **Assist-based scoring** and **no opponent** (see Environment).
- **Messages carry sightings, the learned part is when to send** (a binary gate) instead of choosing among
  attack/defend/pass/rotate tokens: interchangeable tokens carried no content.
- **Message credit**: counterfactual rollouts (or Monte-Carlo team return) instead of DQN on the team reward.
- **Parameter sharing** between the two learners (decentralised execution kept).
- **Rule and fixed-schedule senders** added to separate the value of information from learning when to send.
- **Short episodes / timed matches and a small view radius**, where information has value; a memory of
  past sightings so that sparse messages are not forgotten after 10 steps.
