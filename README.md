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

# Run smoke tests
python tests/test_env.py

# Render a hardcoded ceiling-bounce and goal demonstration
python -m experiments.render_goal_demo

# Train one communication setting (A/B1-B5/C/D)
python -m experiments.train --setting B3 --episodes 1000

# Run all settings at both visibility levels across five seeds
python -m experiments.run_all --episodes 1000

# Plot results
python utils/plot_results.py --dir checkpoints --save
```

---

## Environment

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_agents` | 2 | Cooperative team size |
| `obs_frac` | 1.0 | Observation radius (1.0 = full, 0.25 = very limited) |
| `comm_budget` | None | Max messages per agent per episode (None = unlimited) |
| `comm_delay` | 0 | Steps before a sent message is delivered |
| `comm_every` | 1 | Send only every K steps |
| `max_steps` | 500 | Episode length cap |

**State** (per agent): self pos, teammate pos, ball pos+vel, opponent pos, last received message (one-hot), remaining communication budget fraction.

**Actions**: noop, up, down, left, right, boost.

**Messages**: none, attack, defend, pass, rotate.

**Task reward** (shared): +100 goal, -100 concede, +0.1 proximity to ball. Agents train on this plus discounted potential-based shaping toward the opponent goal (scale 0.01); histories and evaluation report task reward separately from shaped reward.

The ASCII demo above uses upward actions to push the ball toward the top boundary. It shows the ball bouncing and the score staying 0–0 when outside the goal, then scoring 1–0 when aligned with the goal.

---

## Communication settings

| Setting | Description |
|---------|-------------|
| **A** | No communication |
| **B1** | Budget of 5 messages / episode |
| **B2** | Budget of 10 messages / episode |
| **B3** | Budget of 20 messages / episode |
| **B4** | Budget of 50 messages / episode |
| **B5** | Budget of 100 messages / episode |
| **C** | Communication every 5 timesteps |
| **D** | Unlimited communication |

---

## Experiments (from the project design)

| # | Variable | Settings |
|---|----------|---------|
| 1 | Communication budget | A, B1-B5, D at obs_frac = 1.0 / 0.25, evaluated over five seeds |
| 2 | Communication delay | 0 / 1 / 2 / 5 steps |
| 3 | Partial observability | obs_frac = 1.0 / 0.75 / 0.5 / 0.25 |
| 4 | Team size | n_agents = 2 / 3 / 4 |

Training logs greedy evaluation reward and goal rates every 100 episodes (20 evaluation episodes by default); evaluation transitions are never added to replay. The full sweep uses seeds 42, 123, 7, 99, and 17. Override with `--seeds 42 123 7` or adjust evaluation using `--eval_every` and `--n_eval_episodes`.

Training uses 200-step episodes, an easy curriculum for the first 500 episodes (ball near goal, stationary opponent, wider goal), then full difficulty. Greedy evaluation always uses the full environment and 500-step episodes. Goal/concede transitions are oversampled during DQN updates.
