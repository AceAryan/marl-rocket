# Pilot results and decisions

Seeds 100-102. Scores: goals per episode (%), best validation snapshot on 200 held-out episodes (match mode: 100). Rules were fixed before the pilot ran (README, Pilot).

## P1: parameter sharing (single-goal, rho 0.075)

| Setting | Shared median | Independent median | Shared AUC | Independent AUC |
|---|---|---|---|---|
| A | 57.5 | 56.5 | 36.0 | 36.8 |
| Finf | 83.0 | 89.0 | 67.7 | 69.7 |

Decision: **independent learners** (sharing is kept unless it is > 2 pp worse in median score or AUC for A or Finf).

## P2: regime

| Mode | rho | A median | Finf median | Relative gap | A >= 25% of Finf | Collapsed Finf seeds | Eligible |
|---|---|---|---|---|---|---|---|
| match | 0.06 | 0.0 | 3.0 | +inf | no | 1 | no |
| match | 0.075 | 3.0 | 16.0 | +4.33 | no | 0 | no |
| match | 0.1 | 41.0 | 449.0 | +9.95 | no | 0 | no |
| single | 0.06 | 26.0 | 71.0 | +1.73 | yes | 0 | yes |
| single | 0.075 | 57.5 | 83.0 | +0.44 | yes | 0 | yes |
| single | 0.1 | 87.0 | 97.5 | +0.12 | yes | 0 | yes |

Decision: **single mode at rho = 0.06** (largest relative gap among eligible regimes; single-goal kept unless match mode's gap is >= 1.5x larger).

## P3: senders (single, rho 0.075)

| Setting | Seeds | Median | Last-policy median | Msgs/agent |
|---|---|---|---|---|
| A | 3 | 57.5 | 39.0 | 0.0 |
| Finf | 3 | 83.0 | 80.5 | 46.8 |
| R4 | 3 | 83.0 | 68.5 | 2.0 |
| Rinf | 3 | 81.5 | 76.0 | 2.2 |
| L4 | 3 | 69.0 | 56.5 | 3.1 |
| L4-team | 3 | 64.0 | 57.5 | 3.3 |
| Linf | 3 | 79.0 | 78.0 | 6.0 |

Decision: credit = **rollout** (higher median at L4).

## Training length and stability

Episodes: **15000** (smallest of (10000, 15000) where the median validation curves of A, Finf and L4 move < 2 pp over the last 20% of training; pilot ran 15000).
Runs (among the 45 with a selected score >= 20) whose last policy scores under half of their selected snapshot: 7% -> LR decay **off** (rule: on if > 10%).

Messages per agent per episode used by the unlimited senders: Finf 52.6.

```json
{
 "share": false,
 "mode": "single",
 "rho": 0.06,
 "credit": "rollout",
 "episodes": 15000,
 "lr_decay": false
}
```
