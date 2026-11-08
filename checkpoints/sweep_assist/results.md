# Results

Final greedy evaluation on 200 held-out episodes per seed. Goal rate = share of episodes with an assisted goal. Δ and p: Welch t-test against A at the same observation radius (* p<0.05, ** p<0.01). Steps to goal = mean episode length (200 = no goal). *Goal % silenced* = the same policy evaluated with the channel muted (how much it relies on messages).

## RQ1: message budget (rho = 0.15)

| Setting | Seeds | Goal rate % (95% CI) | Δ vs A (pp) | p | Steps to goal | Msgs/agent | Goal % silenced |
|---|---|---|---|---|---|---|---|
| A: no communication | 5 | 90.6 ± 10.3 | - | - | 67 | 0.0 | - |
| D5: learned, budget 5 | 5 | 80.0 ± 16.1 | -10.6 | 0.168 | 103 | 5.0 | 16.6 |
| D10: learned, budget 10 | 5 | 74.9 ± 12.7 | -15.7 * | 0.030 | 112 | 10.0 | 13.6 |
| D: learned, budget 20 | 5 | 78.5 ± 14.5 | -12.1 | 0.100 | 103 | 19.3 | 9.7 |
| D50: learned, budget 50 | 5 | 72.4 ± 4.1 | -18.2 ** | 0.005 | 110 | 46.1 | 13.0 |
| C: learned, every 5 steps | 5 | 76.2 ± 21.6 | -14.4 | 0.148 | 105 | 17.4 | 17.3 |
| B: learned, unlimited | 5 | 96.2 ± 4.8 | +5.6 | 0.222 | 61 | 53.2 | 12.3 |

## RQ1: message budget (full view, rho = 1.0)

| Setting | Seeds | Goal rate % (95% CI) | Δ vs A (pp) | p | Steps to goal | Msgs/agent | Goal % silenced |
|---|---|---|---|---|---|---|---|
| A: no communication | 5 | 97.9 ± 1.4 | - | - | 50 | 0.0 | - |
| D5: learned, budget 5 | 5 | 90.1 ± 3.0 | -7.8 ** | 0.001 | 82 | 4.9 | 57.0 |
| D10: learned, budget 10 | 5 | 97.0 ± 2.8 | -0.9 | 0.458 | 62 | 10.0 | 48.7 |
| D: learned, budget 20 | 5 | 97.8 ± 0.9 | -0.1 | 0.875 | 59 | 19.7 | 49.3 |
| D50: learned, budget 50 | 5 | 68.2 ± 44.8 | -29.7 | 0.140 | 120 | 46.9 | 5.2 |
| C: learned, every 5 steps | 5 | 96.1 ± 3.9 | -1.8 | 0.281 | 63 | 11.1 | 42.1 |
| B: learned, unlimited | 5 | 97.8 ± 1.8 | -0.1 | 0.908 | 55 | 49.2 | 9.0 |

## RQ2: message delay (rho = 0.15)

| Setting | Seeds | Goal rate % (95% CI) | Δ vs A (pp) | p | Steps to goal | Msgs/agent | Goal % silenced |
|---|---|---|---|---|---|---|---|
| A: no communication | 5 | 90.6 ± 10.3 | - | - | 67 | 0.0 | - |
| B: learned, unlimited | 5 | 96.2 ± 4.8 | +5.6 | 0.222 | 61 | 53.2 | 12.3 |
| B: learned, unlimited | 5 | 74.6 ± 50.1 | -16.0 | 0.431 | 91 | 81.9 | 7.7 |
| B: learned, unlimited | 5 | 89.1 ± 14.1 | -1.5 | 0.818 | 73 | 59.8 | 14.5 |
| B: learned, unlimited | 5 | 85.1 ± 11.2 | -5.5 | 0.344 | 84 | 69.0 | 12.5 |

## RQ3: observability

| rho | A: no communication | B: learned, unlimited | Finf: fixed, every step |
|---|---|---|---|
| 1 | 97.9 ± 1.4 | 97.8 ± 1.8 | - |
| 0.25 | 96.1 ± 4.5 | 94.4 ± 5.5 | - |
| 0.15 | 90.6 ± 10.3 | 96.2 ± 4.8 | - |
| 0.1 | 78.2 ± 4.3 | 64.2 ± 13.4 * | - |

## Message timing (greedy, rho = 0.15)

| Policy | Seeds | Median send step | % sent in first 10 steps | % sent while seeing ball |
|---|---|---|---|---|
| Learned D, event-triggered | 5 | 21 | 10 | 100 |
