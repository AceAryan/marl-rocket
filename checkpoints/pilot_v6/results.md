# Results

Main regime: **single-goal episodes, rho = 0.075**. Primary metric: goals per episode on 200 held-out episodes (single-goal mode: share of episodes with an assisted goal), in %, for the policy snapshot with the best validation score; *Last policy* is the end-of-training policy on the same episodes (*Collapsed* = seeds whose last policy scores under half of their selected snapshot). Median over seeds with interquartile range, and mean with a 95% t-interval. p: two-sided Mann-Whitney U, **Holm-corrected within each table** (* p<0.05, ** p<0.01; no stars with fewer than 5 seeds per group). AUC: mean validation score over training (learning speed). *Shuffled-content drop*: score minus the score when messages are delivered at the same times but carry another episode's content. Time to goal: first goal step among scoring episodes.

## Reference teams (no learning)

| Reference team | Goals % | Time to goal |
|---|---|---|
| random | 0.0 | - |
| scripted | 60.5 | 50 |
| scripted_comm | 52.0 | 56 |

## RQ1: message budget

### Learned gate (budget = messages per agent per episode)

| Configuration | Seeds | Goals % median [IQR] | Mean ± 95% CI | Δ vs A | p vs A | Δ vs unlimited | p vs unlimited | Last policy median | Collapsed (last) | Msgs/agent | Time to goal | AUC | Shuffled-content drop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: no communication | 3 | 57.5 [57-60] | 59.2 ± 7.2 | - | - | - | - | 39.0 | 1 | 0.0 | 40 | 36.0 | - |
| L4 | 3 | 69.0 [62-72] | 66.7 ± 24.1 | +11.5 | 0.600 | -10.0 | 0.400 | 56.5 | 0 | 3.1 | 42 | 39.8 | +28.0 |
| Linf | 3 | 79.0 [77-83] | 80.0 ± 15.1 | +21.5 | 0.300 | - | - | 78.0 | 0 | 6.0 | 40 | 55.3 | +34.5 |

Trend over finite budgets (incl. 0): Spearman rho = +0.30, p = 0.568 (n = 6 runs).
Saturation fit y(B) = y_inf - (y_inf - y0)·exp(-B/beta): y0 = 59.2, y_inf = 80.0, **B90 = 20.7** messages (95% seed-bootstrap CI 6.1-1151.3); smallest tested budget whose median reaches 90% of the gain: **inf** (CI 4--).

### Rule: send on sight (budget = messages per agent per episode)

| Configuration | Seeds | Goals % median [IQR] | Mean ± 95% CI | Δ vs A | p vs A | Δ vs unlimited | p vs unlimited | Last policy median | Collapsed (last) | Msgs/agent | Time to goal | AUC | Shuffled-content drop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: no communication | 3 | 57.5 [57-60] | 59.2 ± 7.2 | - | - | - | - | 39.0 | 1 | 0.0 | 40 | 36.0 | - |
| R4 | 3 | 83.0 [76-84] | 78.8 ± 21.2 | +25.5 | 0.300 | +1.5 | 0.700 | 68.5 | 0 | 2.0 | 43 | 52.5 | +26.0 |
| Rinf | 3 | 81.5 [80-82] | 80.8 ± 5.2 | +24.0 | 0.300 | - | - | 76.0 | 0 | 2.2 | 43 | 49.7 | +22.5 |

Trend over finite budgets (incl. 0): Spearman rho = +0.89, p = 0.0171 (n = 6 runs).
Saturation fit y(B) = y_inf - (y_inf - y0)·exp(-B/beta): y0 = 59.2, y_inf = 80.8, **B90 = 3.8** messages (95% seed-bootstrap CI 0.1-14.3); smallest tested budget whose median reaches 90% of the gain: **4** (CI 4--).

### Fixed schedule (budget = messages per agent per episode)

| Configuration | Seeds | Goals % median [IQR] | Mean ± 95% CI | Δ vs A | p vs A | Δ vs unlimited | p vs unlimited | Last policy median | Collapsed (last) | Msgs/agent | Time to goal | AUC | Shuffled-content drop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: no communication | 3 | 57.5 [57-60] | 59.2 ± 7.2 | - | - | - | - | 39.0 | 1 | 0.0 | 40 | 36.0 | - |
| Finf | 3 | 83.0 [82-86] | 84.5 ± 9.9 | +25.5 | 0.100 | - | - | 80.5 | 0 | 46.8 | 41 | 67.8 | +55.5 |

Trend over finite budgets (incl. 0): Spearman rho = +0.00, p = 1 (n = 3 runs).
Saturation fit y(B) = y_inf - (y_inf - y0)·exp(-B/beta): y0 = 59.2, y_inf = 84.5, **B90 = 0.1** messages (95% seed-bootstrap CI 0.1-0.1); smallest tested budget whose median reaches 90% of the gain: **inf** (CI ---).

## RQ3: observability

| rho | A median [IQR] | Linf median [IQR] | Rinf median [IQR] | Linf - A (95% CI) | Rinf - A (95% CI) |
|---|---|---|---|---|---|
| 0.1 | 87.0 [85-88] (n=3) | - | - | - | - |
| 0.075 | 57.5 [57-60] (n=3) | 79.0 [77-83] (n=3) | 81.5 [80-82] (n=3) | +21.5 (+12.0 to +29.0) | +24.0 (+16.0 to +25.0) |
| 0.06 | 26.0 [20-26] (n=3) | - | - | - | - |


## Other learned-sender variants

| Configuration | Seeds | Goals % median [IQR] | Δ vs L4 | p |
|---|---|---|---|---|
| L4 | 3 | 69.0 [62-72] | - | - |
| L4-team | 3 | 64.0 [57-66] | -5.0 | 0.500 |
| Linf | 3 | 79.0 [77-83] | +10.0 | 0.200 |
