# Results

Each run keeps the policy snapshot with the best goal rate on the periodic validation episodes; it is then scored on 200 separate held-out episodes. Goal rate = share of episodes with an assisted goal. Δ and p: Welch t-test against A at the same observation radius (* p<0.05, ** p<0.01). Steps to goal = mean episode length (80 = no goal). *Goal % silenced* = the same policy evaluated with the channel muted (how much it relies on messages).

## RQ1: message budget (rho = 0.15)

| Setting | Seeds | Goal rate % (95% CI) | Δ vs A (pp) | p | Steps to goal | Msgs/agent | Goal % silenced |
|---|---|---|---|---|---|---|---|
| A: no communication | 2 | 85.2 ± 3.2 | - | - | 49 | 0.0 | - |
| D2: learned, budget 2 | 2 | 47.0 ± 57.2 | -38.2 | 0.074 | 66 | 1.9 | 45.5 |
| D5: learned, budget 5 | 2 | 58.0 ± 127.1 | -27.2 | 0.224 | 63 | 4.1 | 29.5 |
| D: learned, budget 20 | 2 | 38.2 ± 346.2 | -47.0 | 0.334 | 68 | 12.9 | 28.2 |
| B: learned, unlimited | 2 | 80.5 ± 25.4 | -4.8 | 0.249 | 53 | 21.8 | 35.2 |
| F5: fixed, every 16 steps | 2 | 86.2 ± 28.6 | +1.0 | 0.734 | 49 | 2.6 | 74.0 |
| Finf: fixed, every step | 2 | 90.8 ± 41.3 | +5.5 | 0.338 | 46 | 45.5 | 0.8 |

## RQ1: message budget (rho = 0.1)

| Setting | Seeds | Goal rate % (95% CI) | Δ vs A (pp) | p | Steps to goal | Msgs/agent | Goal % silenced |
|---|---|---|---|---|---|---|---|
| A: no communication | 2 | 82.2 ± 22.2 | - | - | 48 | 0.0 | - |
| D5: learned, budget 5 | 2 | 29.8 ± 28.6 | -52.5 ** | 0.004 | 71 | 3.7 | 19.0 |
| B: learned, unlimited | 2 | 23.5 ± 44.5 | -58.8 * | 0.014 | 71 | 20.7 | 6.8 |

## RQ2: message delay (rho = 0.15)

| Setting | Seeds | Goal rate % (95% CI) | Δ vs A (pp) | p | Steps to goal | Msgs/agent | Goal % silenced |
|---|---|---|---|---|---|---|---|
| A: no communication | 2 | 85.2 ± 3.2 | - | - | 49 | 0.0 | - |
| B: learned, unlimited, delay 0 | 2 | 80.5 ± 25.4 | -4.8 | 0.249 | 53 | 21.8 | 35.2 |
| Finf: fixed, every step, delay 0 | 2 | 90.8 ± 41.3 | +5.5 | 0.338 | 46 | 45.5 | 0.8 |

## RQ3: observability

| rho | A: no communication | B: learned, unlimited | Finf: fixed, every step |
|---|---|---|---|
| 0.15 | 85.2 ± 3.2 | 80.5 ± 25.4 | 90.8 ± 41.3 |
| 0.1 | 82.2 ± 22.2 | 23.5 ± 44.5 * | - |

## Message timing (greedy, rho = 0.15)

| Policy | Seeds | Median send step | % sent in first 10 steps | % sent while seeing ball |
|---|---|---|---|---|
| Learned D, event-triggered | 2 | 31 | 7 | 100 |
