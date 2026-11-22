# Results

Each run keeps the policy snapshot with the best goal rate on the periodic validation episodes; it is then scored on 200 separate held-out episodes. Goal rate = share of episodes with an assisted goal; median over seeds with its interquartile range, and the mean with a 95% t-interval. Δ: difference in medians from A at the same observation radius; p: two-sided Mann-Whitney U test against A (* p<0.05, ** p<0.01). Collapsed = seeds below 50% goals. Steps to goal = mean episode length (80 = no goal). *Goal % silenced* = the same policy evaluated with the channel muted (how much it relies on messages).

## RQ1: message budget (rho = 0.15)

| Setting | Seeds | Goal % median [IQR] | Mean ± 95% CI | Δ median vs A (pp) | p | Collapsed | Steps to goal | Msgs/agent | Goal % silenced |
|---|---|---|---|---|---|---|---|---|---|
| A: no communication | 5 | 96.5 [88-98] | 93.9 ± 7.1 | - | - | 0 | 44 | 0.0 | - |
| D2: learned, budget 2 | 5 | 91.5 [90-95] | 89.6 ± 10.1 | -5.0 | 0.421 | 0 | 45 | 1.7 | 88.5 |
| D5: learned, budget 5 | 5 | 84.0 [76-84] | 69.6 ± 42.0 | -12.5 * | 0.032 | 1 | 53 | 3.3 | 56.0 |
| D10: learned, budget 10 | 5 | 91.0 [88-91] | 88.9 ± 6.6 | -5.5 | 0.349 | 0 | 47 | 6.3 | 64.5 |
| D: learned, budget 20 | 5 | 93.0 [92-97] | 94.1 ± 5.2 | -3.5 | 0.897 | 0 | 43 | 10.5 | 63.0 |
| C: learned, every 5 steps | 5 | 93.5 [88-94] | 92.3 ± 4.6 | -3.0 | 0.683 | 0 | 44 | 3.8 | 54.0 |
| B: learned, unlimited | 5 | 94.0 [92-95] | 94.0 ± 2.7 | -2.5 | 0.841 | 0 | 45 | 5.8 | 63.5 |
| R2: rule, budget 2 | 5 | 92.5 [92-94] | 91.6 ± 6.6 | -4.0 | 0.381 | 0 | 46 | 2.0 | 7.0 |
| R5: rule, budget 5 | 5 | 96.5 [93-96] | 95.0 ± 2.6 | +0.0 | 0.841 | 0 | 42 | 5.0 | 42.5 |
| R10: rule, budget 10 | 5 | 93.0 [92-94] | 86.3 ± 19.3 | -3.5 | 0.421 | 0 | 48 | 9.9 | 25.0 |
| R20: rule, budget 20 | 5 | 93.0 [92-96] | 93.3 ± 3.4 | -3.5 | 0.738 | 0 | 44 | 18.4 | 51.0 |
| Rinf: rule, unlimited | 5 | 98.0 [96-98] | 97.0 ± 3.2 | +1.5 | 0.524 | 0 | 40 | 21.6 | 26.0 |
| F2: fixed, every 40 steps | 5 | 88.0 [88-92] | 89.6 ± 4.1 | -8.5 | 0.222 | 0 | 46 | 0.8 | 91.0 |
| F5: fixed, every 16 steps | 5 | 90.5 [90-97] | 92.8 ± 5.4 | -6.0 | 0.968 | 0 | 44 | 2.4 | 76.5 |
| F10: fixed, every 8 steps | 5 | 92.5 [88-95] | 82.0 ± 31.1 | -4.0 | 0.421 | 1 | 50 | 5.8 | 5.0 |
| F20: fixed, every 4 steps | 5 | 97.0 [96-98] | 96.3 ± 3.7 | +0.5 | 0.690 | 0 | 42 | 10.1 | 3.0 |
| Finf: fixed, every step | 5 | 91.0 [90-94] | 92.3 ± 4.2 | -5.5 | 0.730 | 0 | 43 | 43.1 | 1.5 |

## RQ1: message budget (rho = 0.1)

| Setting | Seeds | Goal % median [IQR] | Mean ± 95% CI | Δ median vs A (pp) | p | Collapsed | Steps to goal | Msgs/agent | Goal % silenced |
|---|---|---|---|---|---|---|---|---|---|
| A: no communication | 5 | 91.5 [89-92] | 90.9 ± 4.4 | - | - | 0 | 43 | 0.0 | - |
| D2: learned, budget 2 | 5 | 82.0 [81-86] | 83.4 ± 5.0 | -9.5 * | 0.024 | 0 | 52 | 1.7 | 80.5 |
| D5: learned, budget 5 | 5 | 90.0 [84-91] | 86.6 ± 7.9 | -1.5 | 0.317 | 0 | 49 | 3.5 | 37.5 |
| D10: learned, budget 10 | 5 | 92.0 [90-95] | 84.2 ± 25.0 | +0.5 | 0.730 | 1 | 46 | 5.1 | 23.5 |
| D: learned, budget 20 | 5 | 93.0 [91-93] | 91.9 ± 2.3 | +1.5 | 0.556 | 0 | 44 | 7.7 | 24.0 |
| C: learned, every 5 steps | 5 | 90.5 [89-92] | 88.8 ± 6.3 | -1.0 | 0.889 | 0 | 49 | 3.3 | 58.0 |
| B: learned, unlimited | 5 | 88.0 [9-94] | 57.4 ± 59.6 | -3.5 | 0.516 | 2 | 58 | 9.5 | 16.5 |
| R2: rule, budget 2 | 5 | 88.5 [80-92] | 85.9 ± 8.8 | -3.0 | 0.413 | 0 | 49 | 2.0 | 15.0 |
| R5: rule, budget 5 | 5 | 89.0 [88-91] | 82.9 ± 20.2 | -2.5 | 0.460 | 0 | 53 | 4.9 | 9.5 |
| R10: rule, budget 10 | 5 | 91.5 [86-92] | 89.4 ± 6.1 | +0.0 | 0.833 | 0 | 43 | 9.5 | 39.5 |
| R20: rule, budget 20 | 5 | 93.0 [87-94] | 91.3 ± 6.2 | +1.5 | 0.794 | 0 | 44 | 14.9 | 22.5 |
| Rinf: rule, unlimited | 5 | 94.0 [92-96] | 92.2 ± 7.6 | +2.5 | 0.468 | 0 | 43 | 18.9 | 14.0 |
| F2: fixed, every 40 steps | 5 | 87.5 [78-90] | 82.4 ± 13.9 | -4.0 | 0.135 | 0 | 48 | 0.8 | 89.5 |
| F5: fixed, every 16 steps | 5 | 80.5 [76-89] | 81.2 ± 13.2 | -11.0 | 0.159 | 0 | 49 | 2.7 | 50.0 |
| F10: fixed, every 8 steps | 5 | 93.0 [92-94] | 92.5 ± 3.6 | +1.5 | 0.540 | 0 | 44 | 5.0 | 5.0 |
| F20: fixed, every 4 steps | 5 | 90.5 [88-94] | 91.6 ± 4.7 | -1.0 | 0.968 | 0 | 43 | 10.5 | 4.0 |
| Finf: fixed, every step | 5 | 95.0 [94-98] | 94.2 ± 6.6 | +3.5 | 0.262 | 0 | 44 | 43.9 | 0.5 |

## RQ1: message budget (full view, rho = 1.0)

| Setting | Seeds | Goal % median [IQR] | Mean ± 95% CI | Δ median vs A (pp) | p | Collapsed | Steps to goal | Msgs/agent | Goal % silenced |
|---|---|---|---|---|---|---|---|---|---|
| A: no communication | 5 | 97.0 [92-98] | 95.6 ± 4.5 | - | - | 0 | 40 | 0.0 | - |
| B: learned, unlimited | 5 | 92.5 [91-97] | 82.8 ± 33.0 | -4.5 | 0.357 | 1 | 48 | 5.8 | 77.0 |
| Rinf: rule, unlimited | 5 | 86.0 [82-94] | 85.9 ± 11.5 | -11.0 | 0.095 | 0 | 50 | 49.9 | 1.5 |
| Finf: fixed, every step | 5 | 86.0 [82-94] | 85.9 ± 11.5 | -11.0 | 0.095 | 0 | 50 | 49.9 | 1.5 |

## RQ2: message delay (rho = 0.15)

| Setting | Seeds | Goal % median [IQR] | Mean ± 95% CI | Δ median vs A (pp) | p | Collapsed | Steps to goal | Msgs/agent | Goal % silenced |
|---|---|---|---|---|---|---|---|---|---|
| A: no communication | 5 | 96.5 [88-98] | 93.9 ± 7.1 | - | - | 0 | 44 | 0.0 | - |
| B: learned, unlimited, delay 0 | 5 | 94.0 [92-95] | 94.0 ± 2.7 | -2.5 | 0.841 | 0 | 45 | 5.8 | 63.5 |
| B: learned, unlimited, delay 1 | 5 | 91.5 [90-94] | 92.1 ± 4.2 | -5.0 | 0.690 | 0 | 44 | 5.1 | 55.5 |
| B: learned, unlimited, delay 2 | 5 | 88.5 [88-92] | 85.6 ± 15.0 | -8.0 | 0.238 | 0 | 48 | 7.9 | 66.0 |
| B: learned, unlimited, delay 5 | 5 | 91.0 [89-92] | 90.0 ± 6.8 | -5.5 | 0.452 | 0 | 45 | 3.8 | 84.5 |
| Rinf: rule, unlimited, delay 0 | 5 | 98.0 [96-98] | 97.0 ± 3.2 | +1.5 | 0.524 | 0 | 40 | 21.6 | 26.0 |
| Rinf: rule, unlimited, delay 1 | 5 | 94.0 [94-96] | 85.0 ± 28.1 | -2.5 | 0.460 | 1 | 46 | 28.0 | 33.0 |
| Rinf: rule, unlimited, delay 2 | 5 | 91.0 [88-92] | 89.0 ± 8.2 | -5.5 | 0.333 | 0 | 45 | 27.2 | 32.5 |
| Rinf: rule, unlimited, delay 5 | 5 | 88.5 [88-90] | 88.3 ± 4.8 | -8.0 | 0.238 | 0 | 47 | 27.7 | 23.0 |
| Finf: fixed, every step, delay 0 | 5 | 91.0 [90-94] | 92.3 ± 4.2 | -5.5 | 0.730 | 0 | 43 | 43.1 | 1.5 |
| Finf: fixed, every step, delay 1 | 5 | 94.0 [94-94] | 94.0 ± 2.0 | -2.5 | 0.651 | 0 | 44 | 43.6 | 2.5 |
| Finf: fixed, every step, delay 2 | 5 | 94.5 [84-94] | 86.6 ± 17.4 | -2.0 | 0.246 | 0 | 49 | 48.6 | 1.5 |
| Finf: fixed, every step, delay 5 | 5 | 84.0 [39-92] | 68.7 ± 36.9 | -12.5 | 0.095 | 2 | 56 | 56.3 | 1.0 |

## RQ3: observability

| rho | A: no communication | B: learned, unlimited | Rinf: rule, unlimited | Finf: fixed, every step |
|---|---|---|---|---|
| 1 | 97.0 [92-98] | 92.5 [91-97] | 86.0 [82-94] | 86.0 [82-94] |
| 0.25 | 98.0 [94-98] | 89.0 [88-94] * | 86.0 [85-86] ** | 89.0 [87-93] * |
| 0.15 | 96.5 [88-98] | 94.0 [92-95] | 98.0 [96-98] | 91.0 [90-94] |
| 0.1 | 91.5 [89-92] | 88.0 [9-94] | 94.0 [92-96] | 95.0 [94-98] |

## Message timing (greedy, rho = 0.15)

| Policy | Seeds | Median send step | % sent in first 10 steps | % sent while seeing ball |
|---|---|---|---|---|
| Learned D5 | 5 | 22 | 16 | 100 |
| Rule R5 | 5 | 11 | 42 | 100 |
| Fixed F5 | 5 | 31 | 0 | 81 |
