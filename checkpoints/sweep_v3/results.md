# Results

Each run keeps the policy snapshot with the best goal rate on the periodic validation episodes; it is then scored on 200 separate held-out episodes. Goal rate = share of episodes with an assisted goal. Δ and p: Welch t-test against A at the same observation radius (* p<0.05, ** p<0.01). Steps to goal = mean episode length (80 = no goal). *Goal % silenced* = the same policy evaluated with the channel muted (how much it relies on messages).

## RQ1: message budget (rho = 0.15)

| Setting | Seeds | Goal rate % (95% CI) | Δ vs A (pp) | p | Steps to goal | Msgs/agent | Goal % silenced |
|---|---|---|---|---|---|---|---|
| A: no communication | 10 | 83.0 ± 8.3 | - | - | 51 | 0.0 | - |
| D2: learned, budget 2 | 10 | 67.3 ± 9.9 | -15.7 * | 0.014 | 59 | 1.8 | 58.3 |
| D5: learned, budget 5 | 10 | 64.8 ± 12.8 | -18.2 * | 0.017 | 59 | 3.9 | 40.2 |
| D10: learned, budget 10 | 10 | 58.1 ± 10.6 | -24.8 ** | 0.001 | 61 | 6.9 | 17.9 |
| D: learned, budget 20 | 10 | 70.1 ± 15.6 | -12.9 | 0.122 | 56 | 8.4 | 36.5 |
| C: learned, every 5 steps | 10 | 53.1 ± 25.6 | -29.8 * | 0.030 | 61 | 5.6 | 22.9 |
| B: learned, unlimited | 10 | 73.2 ± 11.1 | -9.7 | 0.132 | 55 | 11.8 | 28.6 |
| R2: rule, budget 2 | 10 | 69.7 ± 14.0 | -13.2 | 0.087 | 56 | 2.0 | 11.3 |
| R5: rule, budget 5 | 10 | 75.0 ± 6.1 | -8.0 | 0.098 | 54 | 5.0 | 13.3 |
| R10: rule, budget 10 | 10 | 73.6 ± 13.3 | -9.4 | 0.197 | 55 | 9.9 | 16.6 |
| R20: rule, budget 20 | 10 | 66.2 ± 15.4 | -16.8 * | 0.048 | 57 | 18.9 | 19.6 |
| Rinf: rule, unlimited | 10 | 83.6 ± 7.4 | +0.6 | 0.897 | 49 | 28.7 | 13.1 |
| F2: fixed, every 40 steps | 10 | 82.9 ± 4.1 | -0.0 | 0.990 | 50 | 0.9 | 83.5 |
| F5: fixed, every 16 steps | 10 | 71.8 ± 11.4 | -11.2 | 0.093 | 57 | 3.2 | 51.5 |
| F10: fixed, every 8 steps | 10 | 72.3 ± 10.7 | -10.6 | 0.095 | 56 | 6.7 | 4.9 |
| F20: fixed, every 4 steps | 10 | 81.0 ± 9.4 | -2.0 | 0.724 | 51 | 12.4 | 2.6 |
| Finf: fixed, every step | 10 | 79.3 ± 7.4 | -3.7 | 0.468 | 50 | 50.1 | 0.7 |

## RQ1: message budget (rho = 0.1)

| Setting | Seeds | Goal rate % (95% CI) | Δ vs A (pp) | p | Steps to goal | Msgs/agent | Goal % silenced |
|---|---|---|---|---|---|---|---|
| A: no communication | 10 | 63.9 ± 10.1 | - | - | 58 | 0.0 | - |
| D2: learned, budget 2 | 10 | 47.5 ± 8.4 | -16.3 * | 0.012 | 64 | 1.7 | 37.6 |
| D5: learned, budget 5 | 10 | 57.8 ± 6.5 | -6.1 | 0.268 | 60 | 3.4 | 32.3 |
| D10: learned, budget 10 | 10 | 63.2 ± 6.4 | -0.6 | 0.911 | 58 | 5.6 | 24.2 |
| D: learned, budget 20 | 10 | 64.7 ± 12.4 | +0.9 | 0.906 | 57 | 8.1 | 26.1 |
| C: learned, every 5 steps | 10 | 34.7 ± 14.7 | -29.1 ** | 0.002 | 68 | 4.7 | 14.6 |
| B: learned, unlimited | 10 | 71.8 ± 10.5 | +8.0 | 0.230 | 55 | 10.7 | 27.1 |
| R2: rule, budget 2 | 10 | 63.0 ± 16.9 | -0.8 | 0.928 | 57 | 1.9 | 13.8 |
| R5: rule, budget 5 | 10 | 59.1 ± 7.6 | -4.7 | 0.412 | 60 | 4.8 | 11.0 |
| R10: rule, budget 10 | 10 | 54.5 ± 16.3 | -9.4 | 0.286 | 60 | 9.4 | 12.8 |
| R20: rule, budget 20 | 10 | 61.0 ± 14.0 | -2.8 | 0.718 | 57 | 15.9 | 13.4 |
| Rinf: rule, unlimited | 10 | 68.7 ± 15.8 | +4.8 | 0.570 | 55 | 24.8 | 5.5 |
| F2: fixed, every 40 steps | 10 | 62.2 ± 7.4 | -1.6 | 0.776 | 57 | 1.1 | 63.6 |
| F5: fixed, every 16 steps | 10 | 48.8 ± 11.0 | -15.1 * | 0.035 | 65 | 3.8 | 32.9 |
| F10: fixed, every 8 steps | 10 | 41.0 ± 16.9 | -22.8 * | 0.020 | 67 | 8.2 | 2.1 |
| F20: fixed, every 4 steps | 10 | 61.7 ± 13.4 | -2.1 | 0.776 | 57 | 14.0 | 0.6 |
| Finf: fixed, every step | 10 | 70.4 ± 14.0 | +6.6 | 0.403 | 53 | 53.0 | 0.1 |

## RQ1: message budget (full view, rho = 1.0)

| Setting | Seeds | Goal rate % (95% CI) | Δ vs A (pp) | p | Steps to goal | Msgs/agent | Goal % silenced |
|---|---|---|---|---|---|---|---|
| A: no communication | 10 | 82.9 ± 10.8 | - | - | 51 | 0.0 | - |
| B: learned, unlimited | 10 | 79.6 ± 13.1 | -3.3 | 0.666 | 51 | 10.3 | 17.4 |
| Rinf: rule, unlimited | 10 | 74.5 ± 9.7 | -8.4 | 0.207 | 54 | 54.1 | 0.8 |
| Finf: fixed, every step | 10 | 74.5 ± 9.7 | -8.4 | 0.207 | 54 | 54.1 | 0.8 |

## RQ2: message delay (rho = 0.15)

| Setting | Seeds | Goal rate % (95% CI) | Δ vs A (pp) | p | Steps to goal | Msgs/agent | Goal % silenced |
|---|---|---|---|---|---|---|---|
| A: no communication | 10 | 83.0 ± 8.3 | - | - | 51 | 0.0 | - |
| B: learned, unlimited, delay 0 | 10 | 73.2 ± 11.1 | -9.7 | 0.132 | 55 | 11.8 | 28.6 |
| B: learned, unlimited, delay 1 | 10 | 63.6 ± 18.4 | -19.3 | 0.051 | 58 | 11.2 | 23.9 |
| B: learned, unlimited, delay 2 | 10 | 62.9 ± 19.6 | -20.1 | 0.054 | 59 | 12.7 | 28.0 |
| B: learned, unlimited, delay 5 | 10 | 71.2 ± 9.8 | -11.7 | 0.054 | 54 | 9.4 | 47.7 |
| Rinf: rule, unlimited, delay 0 | 10 | 83.6 ± 7.4 | +0.6 | 0.897 | 49 | 28.7 | 13.1 |
| Rinf: rule, unlimited, delay 1 | 10 | 61.7 ± 21.1 | -21.2 | 0.056 | 58 | 37.6 | 6.2 |
| Rinf: rule, unlimited, delay 2 | 10 | 68.8 ± 16.5 | -14.2 | 0.106 | 56 | 35.4 | 8.7 |
| Rinf: rule, unlimited, delay 5 | 10 | 71.2 ± 12.4 | -11.7 | 0.096 | 54 | 34.7 | 9.8 |
| Finf: fixed, every step, delay 0 | 10 | 79.3 ± 7.4 | -3.7 | 0.468 | 50 | 50.1 | 0.7 |
| Finf: fixed, every step, delay 1 | 10 | 80.2 ± 11.9 | -2.7 | 0.680 | 51 | 51.1 | 0.7 |
| Finf: fixed, every step, delay 2 | 10 | 70.0 ± 18.5 | -12.9 | 0.174 | 55 | 54.7 | 0.8 |
| Finf: fixed, every step, delay 5 | 10 | 65.5 ± 10.8 | -17.4 * | 0.010 | 58 | 58.3 | 0.8 |

## RQ3: observability

| rho | A: no communication | B: learned, unlimited | Rinf: rule, unlimited | Finf: fixed, every step |
|---|---|---|---|---|
| 1 | 82.9 ± 10.8 | 79.6 ± 13.1 | 74.5 ± 9.7 | 74.5 ± 9.7 |
| 0.25 | 91.0 ± 2.3 | 87.8 ± 4.6 | 75.9 ± 15.7 | 69.3 ± 20.1 * |
| 0.15 | 83.0 ± 8.3 | 73.2 ± 11.1 | 83.6 ± 7.4 | 79.3 ± 7.4 |
| 0.1 | 63.9 ± 10.1 | 71.8 ± 10.5 | 68.7 ± 15.8 | 70.4 ± 14.0 |

## Message timing (greedy, rho = 0.15)

| Policy | Seeds | Median send step | % sent in first 10 steps | % sent while seeing ball |
|---|---|---|---|---|
| Learned D5 | 10 | 28 | 11 | 100 |
| Rule R5 | 10 | 11 | 41 | 100 |
| Fixed F5 | 10 | 31 | 0 | 84 |
