# Results

Each run keeps the policy snapshot with the best goal rate on the periodic validation episodes; it is then scored on 200 separate held-out episodes. Goal rate = share of episodes with an assisted goal. Δ and p: Welch t-test against A at the same observation radius (* p<0.05, ** p<0.01). Steps to goal = mean episode length (200 = no goal). *Goal % silenced* = the same policy evaluated with the channel muted (how much it relies on messages).

## RQ1: message budget (rho = 0.15)

| Setting | Seeds | Goal rate % (95% CI) | Δ vs A (pp) | p | Steps to goal | Msgs/agent | Goal % silenced |
|---|---|---|---|---|---|---|---|
| A: no communication | 3 | 92.0 ± 7.6 | - | - | 68 | 0.0 | - |
| D5: learned, budget 5 | 3 | 78.8 ± 23.1 | -13.2 | 0.123 | 101 | 4.7 | 54.0 |
| D: learned, budget 20 | 3 | 78.2 ± 8.0 | -13.8 ** | 0.006 | 108 | 18.3 | 33.0 |
| B: learned, unlimited | 3 | 93.2 ± 12.2 | +1.2 | 0.747 | 66 | 37.1 | 58.7 |
| F20: fixed, every 10 steps | 3 | 88.0 ± 12.2 | -4.0 | 0.309 | 79 | 7.5 | 11.5 |

## RQ1: message budget (full view, rho = 1.0)

| Setting | Seeds | Goal rate % (95% CI) | Δ vs A (pp) | p | Steps to goal | Msgs/agent | Goal % silenced |
|---|---|---|---|---|---|---|---|
| A: no communication | 3 | 98.0 ± 3.3 | - | - | 49 | 0.0 | - |
| B: learned, unlimited | 3 | 83.5 ± 51.2 | -14.5 | 0.347 | 81 | 68.4 | 7.5 |

## RQ2: message delay (rho = 0.15)

| Setting | Seeds | Goal rate % (95% CI) | Δ vs A (pp) | p | Steps to goal | Msgs/agent | Goal % silenced |
|---|---|---|---|---|---|---|---|
| A: no communication | 3 | 92.0 ± 7.6 | - | - | 68 | 0.0 | - |
| B: learned, unlimited, delay 0 | 3 | 93.2 ± 12.2 | +1.2 | 0.747 | 66 | 37.1 | 58.7 |

## RQ3: observability

| rho | A: no communication | B: learned, unlimited | Finf: fixed, every step |
|---|---|---|---|
| 1 | 98.0 ± 3.3 | 83.5 ± 51.2 | - |
| 0.15 | 92.0 ± 7.6 | 93.2 ± 12.2 | - |

## Message timing (greedy, rho = 0.15)

| Policy | Seeds | Median send step | % sent in first 10 steps | % sent while seeing ball |
|---|---|---|---|---|
| Learned D, before fix | 5 | 12 | 40 | 60 |
| Learned D, event-triggered | 3 | 30 | 3 | 100 |
| Fixed F20 | 3 | 39 | 14 | 65 |
