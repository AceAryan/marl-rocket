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
│   └── dqn_agent.py         # Independent DQN + message head (pure numpy)
├── experiments/
│   ├── train.py             # Single-setting training loop
│   └── run_all.py           # Run all 4 communication settings + summary
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

# Train one communication setting (A/B/C/D)
python -m experiments.train --setting B --episodes 1000

# Run all four settings and compare
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

**State** (per agent): self pos, teammate pos, ball pos+vel, opponent pos, last received message (one-hot).

**Actions**: noop, up, down, left, right, boost.

**Messages**: none, attack, defend, pass, rotate.

**Reward** (shared): +100 goal, -100 concede, +0.5×ball_vy, +0.1 proximity to ball.

---

## Communication settings

| Setting | Description |
|---------|-------------|
| **A** | No communication |
| **B** | Every timestep (unlimited) |
| **C** | Every 5 timesteps |
| **D** | Budget of 20 messages / episode |

---

## Experiments (from the project design)

| # | Variable | Settings |
|---|----------|---------|
| 1 | Communication budget | A vs B vs C vs D |
| 2 | Communication delay | 0 / 1 / 2 / 5 steps |
| 3 | Partial observability | obs_frac = 1.0 / 0.75 / 0.5 / 0.25 |
| 4 | Team size | n_agents = 2 / 3 / 4 |
