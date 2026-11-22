# Results

Main regime: **single-goal episodes, rho = 0.06**. Primary metric: goals per episode on 200 held-out episodes (single-goal mode: share of episodes with an assisted goal), in %, for the policy snapshot with the best validation score; *Last policy* is the end-of-training policy on the same episodes (*Collapsed* = seeds whose last policy scores under half of their selected snapshot). Median over seeds with interquartile range, and mean with a 95% t-interval. p: two-sided Mann-Whitney U, **Holm-corrected within each table** (* p<0.05, ** p<0.01; no stars with fewer than 5 seeds per group). AUC: mean validation score over training (learning speed). *Shuffled-content drop*: score minus the score when messages are delivered at the same times but carry another episode's content. Time to goal: first goal step among scoring episodes.

## Reference teams (no learning)

| Reference team | Goals % | Time to goal |
|---|---|---|
| random | 0.0 | - |
| scripted | 42.5 | 49 |
| scripted_comm | 37.0 | 55 |

## RQ1: message budget

### Learned gate (budget = messages per agent per episode)

| Configuration | Seeds | Goals % median [IQR] | Mean ± 95% CI | Δ vs A | p vs A | Δ vs unlimited | p vs unlimited | Last policy median | Collapsed (last) | Msgs/agent | Time to goal | AUC | Shuffled-content drop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: no communication | 10 | 12.8 [10-15] | 11.9 ± 3.7 | - | - | - | - | 3.2 | 6 | 0.0 | 47 | 6.1 | - |
| L1 | 10 | 14.5 [10-16] | 13.0 ± 3.2 | +1.8 | 1.000 | -31.5 ** | 0.000 | 2.5 | 8 | 0.6 | 43 | 7.3 | +4.8 |
| L2 | 10 | 8.8 [7-11] | 10.2 ± 3.6 | -4.0 | 1.000 | -37.2 ** | 0.000 | 2.2 | 8 | 1.1 | 45 | 5.3 | +1.2 |
| L3 | 10 | 15.0 [10-20] | 16.1 ± 5.3 | +2.2 | 1.000 | -31.0 ** | 0.000 | 7.0 | 8 | 1.5 | 42 | 8.4 | +7.8 |
| L4 | 10 | 18.8 [12-27] | 19.1 ± 7.0 | +6.0 | 1.000 | -27.2 ** | 0.002 | 10.8 | 4 | 1.8 | 41 | 11.6 | +10.2 |
| L6 | 10 | 27.5 [19-33] | 29.6 ± 9.2 | +14.8 ** | 0.002 | -18.5 | 0.098 | 13.5 | 4 | 2.8 | 41 | 16.1 | +15.2 |
| L8 | 10 | 30.0 [26-33] | 29.3 ± 5.7 | +17.2 ** | 0.001 | -16.0 * | 0.035 | 27.5 | 2 | 2.9 | 41 | 17.0 | +17.8 |
| L12 | 10 | 37.2 [28-44] | 36.1 ± 9.2 | +24.5 ** | 0.002 | -8.8 | 0.766 | 29.0 | 1 | 3.7 | 42 | 20.7 | +24.0 |
| L16 | 10 | 44.5 [35-55] | 43.5 ± 11.8 | +31.8 ** | 0.000 | -1.5 | 1.000 | 37.2 | 1 | 4.3 | 43 | 24.5 | +25.8 |
| L24 | 10 | 43.8 [36-52] | 45.0 ± 7.2 | +31.0 ** | 0.000 | -2.2 | 1.000 | 35.2 | 0 | 4.5 | 42 | 23.7 | +21.8 |
| Linf | 10 | 46.0 [40-54] | 45.3 ± 7.5 | +33.2 ** | 0.000 | - | - | 34.8 | 0 | 4.3 | 41 | 24.8 | +28.0 |

Trend over finite budgets (incl. 0): Spearman rho = +0.79, p = 4.21e-22 (n = 100 runs).
Saturation fit y(B) = y_inf - (y_inf - y0)·exp(-B/beta): y0 = 7.7, y_inf = 47.3, **B90 = 21.7** messages (95% seed-bootstrap CI 16.5-29.3); smallest tested budget whose median reaches 90% of the gain: **16** (CI 8-inf).

### Rule: send on sight (budget = messages per agent per episode)

| Configuration | Seeds | Goals % median [IQR] | Mean ± 95% CI | Δ vs A | p vs A | Δ vs unlimited | p vs unlimited | Last policy median | Collapsed (last) | Msgs/agent | Time to goal | AUC | Shuffled-content drop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: no communication | 10 | 12.8 [10-15] | 11.9 ± 3.7 | - | - | - | - | 3.2 | 6 | 0.0 | 47 | 6.1 | - |
| R1 | 10 | 24.2 [17-32] | 24.6 ± 8.0 | +11.5 | 0.050 | +2.0 | 1.000 | 8.2 | 6 | 0.7 | 42 | 12.1 | +8.8 |
| R2 | 10 | 19.2 [15-23] | 19.1 ± 4.4 | +6.5 | 0.129 | -3.0 | 1.000 | 4.0 | 7 | 1.0 | 41 | 10.9 | +10.8 |
| R3 | 10 | 16.5 [12-23] | 19.1 ± 6.2 | +3.8 | 0.906 | -5.8 | 1.000 | 4.0 | 7 | 1.1 | 41 | 11.3 | +8.0 |
| R4 | 10 | 32.0 [22-38] | 30.4 ± 6.6 | +19.2 ** | 0.002 | +9.8 | 0.433 | 19.8 | 2 | 1.3 | 40 | 17.6 | +17.2 |
| R6 | 10 | 22.8 [18-27] | 24.7 ± 7.1 | +10.0 * | 0.019 | +0.5 | 1.000 | 14.0 | 4 | 1.3 | 41 | 13.7 | +13.2 |
| R8 | 10 | 22.5 [21-30] | 26.4 ± 7.9 | +9.8 ** | 0.001 | +0.2 | 1.000 | 8.2 | 6 | 1.3 | 43 | 14.3 | +9.5 |
| R12 | 10 | 19.5 [16-25] | 21.9 ± 6.7 | +6.8 | 0.108 | -2.8 | 1.000 | 7.8 | 8 | 1.3 | 42 | 12.4 | +7.5 |
| R16 | 10 | 24.2 [18-31] | 25.1 ± 6.8 | +11.5 * | 0.027 | +2.0 | 1.000 | 15.2 | 4 | 1.3 | 41 | 15.1 | +12.5 |
| R24 | 10 | 21.5 [16-25] | 22.8 ± 6.5 | +8.8 * | 0.040 | -0.8 | 1.000 | 11.0 | 5 | 1.4 | 42 | 12.7 | +10.5 |
| Rinf | 10 | 22.2 [15-24] | 20.1 ± 6.4 | +9.5 | 0.214 | - | - | 13.0 | 3 | 1.3 | 42 | 12.7 | +11.8 |

Trend over finite budgets (incl. 0): Spearman rho = +0.24, p = 0.014 (n = 100 runs).
Saturation fit y(B) = y_inf - (y_inf - y0)·exp(-B/beta): y0 = 11.9, y_inf = 23.4, **B90 = 0.1** messages (95% seed-bootstrap CI 0.1-4.7); smallest tested budget whose median reaches 90% of the gain: **1** (CI 1-6).

### Fixed schedule (budget = messages per agent per episode)

| Configuration | Seeds | Goals % median [IQR] | Mean ± 95% CI | Δ vs A | p vs A | Δ vs unlimited | p vs unlimited | Last policy median | Collapsed (last) | Msgs/agent | Time to goal | AUC | Shuffled-content drop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: no communication | 10 | 12.8 [10-15] | 11.9 ± 3.7 | - | - | - | - | 3.2 | 6 | 0.0 | 47 | 6.1 | - |
| F2 | 10 | 15.8 [12-25] | 18.1 ± 6.3 | +3.0 | 0.148 | -61.8 ** | 0.000 | 9.0 | 6 | 1.8 | 41 | 9.4 | +11.2 |
| F4 | 10 | 7.5 [5-9] | 7.8 ± 3.0 | -5.2 | 0.100 | -70.0 ** | 0.000 | 1.5 | 10 | 3.9 | 46 | 4.0 | +4.2 |
| F8 | 10 | 21.5 [16-27] | 23.9 ± 7.4 | +8.8 ** | 0.006 | -56.0 ** | 0.000 | 8.2 | 6 | 7.1 | 43 | 13.7 | +16.8 |
| F16 | 10 | 58.0 [56-60] | 57.1 ± 5.3 | +45.2 ** | 0.000 | -19.5 ** | 0.000 | 46.5 | 1 | 11.6 | 42 | 35.5 | +45.5 |
| F26 | 10 | 56.5 [53-60] | 57.5 ± 4.5 | +43.8 ** | 0.000 | -21.0 ** | 0.000 | 52.5 | 0 | 19.3 | 43 | 38.4 | +44.2 |
| Finf | 10 | 77.5 [74-80] | 78.8 ± 4.6 | +64.8 ** | 0.000 | - | - | 64.2 | 0 | 50.7 | 43 | 52.6 | +61.5 |

Trend over finite budgets (incl. 0): Spearman rho = +0.74, p = 9.64e-12 (n = 60 runs).
Saturation fit y(B) = y_inf - (y_inf - y0)·exp(-B/beta): y0 = 6.5, y_inf = 79.5, **B90 = 46.5** messages (95% seed-bootstrap CI 41.5-53.4); smallest tested budget whose median reaches 90% of the gain: **inf** (CI inf-inf).

## Budget awareness (feedback 2)

Learned gate with the budget features (own messages left, teammate's inferred messages left, fraction left) vs the same setting with them hidden.

| Budget | Aware median [IQR] | Blind median [IQR] | Δ (aware - blind) | p (Holm) | Msgs/agent aware / blind | Last-quarter sends aware / blind |
|---|---|---|---|---|---|---|
| L2 | 8.8 [7-11] | 16.2 [9-19] | -7.5 | 0.301 | 1.1 / 1.1 | 3% / 2% |
| L4 | 18.8 [12-27] | 24.2 [16-30] | -5.5 | 0.301 | 1.8 / 2.1 | 1% / 2% |
| L8 | 30.0 [26-33] | 38.2 [28-44] | -8.2 | 0.301 | 2.9 / 3.3 | 1% / 1% |

## RQ2: message delay (rho = 0.06)

**Linf**

| Delay | Seeds | Goals % median [IQR] | Δ vs delay 0 | p (Holm) | Msgs/agent |
|---|---|---|---|---|---|
| 0 | 10 | 46.0 [40-54] | - | - | 4.3 |
| 1 | 10 | 46.2 [37-50] | +0.2 | 0.684 | 4.5 |
| 2 | 10 | 36.5 [29-39] | -9.5 | 0.099 | 4.5 |
| 4 | 10 | 31.2 [23-37] | -14.8 | 0.066 | 4.1 |
| 8 | 10 | 25.0 [18-29] | -21.0 ** | 0.001 | 4.2 |

Trend over delay: Spearman rho = -0.61, p = 2.1e-06.

**Rinf**

| Delay | Seeds | Goals % median [IQR] | Δ vs delay 0 | p (Holm) | Msgs/agent |
|---|---|---|---|---|---|
| 0 | 10 | 22.2 [15-24] | - | - | 1.3 |
| 1 | 10 | 23.8 [21-25] | +1.5 | 1.000 | 1.4 |
| 2 | 10 | 23.5 [19-28] | +1.2 | 1.000 | 1.3 |
| 4 | 10 | 19.0 [17-23] | -3.2 | 1.000 | 1.4 |
| 8 | 10 | 17.2 [15-20] | -5.0 | 1.000 | 1.4 |

Trend over delay: Spearman rho = -0.23, p = 0.104.

## RQ3: observability

| rho | A median [IQR] | Linf median [IQR] | Rinf median [IQR] | Linf - A (95% CI) | Rinf - A (95% CI) |
|---|---|---|---|---|---|
| 1 | 98.5 [98-99] (n=10) | 98.0 [97-98] (n=10) | - | -0.5 (-2.5 to +1.0) | - |
| 0.25 | 98.2 [98-99] (n=10) | 97.0 [94-97] (n=10) | 96.2 [94-97] (n=10) | -1.2 (-4.2 to -0.2) | -2.0 (-5.0 to -0.2) |
| 0.15 | 95.5 [94-96] (n=10) | 96.5 [96-97] (n=10) | 93.0 [92-95] (n=10) | +1.0 (+0.0 to +3.3) | -2.5 (-4.5 to +0.8) |
| 0.1 | 84.5 [83-86] (n=10) | 90.0 [88-92] (n=10) | 87.8 [85-90] (n=10) | +5.5 (+2.0 to +8.5) | +3.2 (-0.8 to +7.0) |
| 0.075 | 60.2 [50-62] (n=10) | 78.5 [76-81] (n=10) | 75.5 [69-78] (n=10) | +18.2 (+14.5 to +30.5) | +15.2 (+7.2 to +27.5) |
| 0.06 | 12.8 [10-15] (n=10) | 46.0 [40-54] (n=10) | 22.2 [15-24] (n=10) | +33.2 (+24.2 to +43.0) | +9.5 (+1.0 to +15.0) |

Linf: slope of the gain against log(rho) = -9.5 pp per log unit (95% bootstrap CI -12.3 to -7.9; negative = communication helps more as view shrinks).
Rinf: slope of the gain against log(rho) = -11.2 pp per log unit (95% bootstrap CI -15.7 to -6.6; negative = communication helps more as view shrinks).

## Saturation at a second radius (rho = 0.1)

### Rule: send on sight (budget = messages per agent per episode)

| Configuration | Seeds | Goals % median [IQR] | Mean ± 95% CI | Δ vs A | p vs A | Δ vs unlimited | p vs unlimited | Last policy median | Collapsed (last) | Msgs/agent | Time to goal | AUC | Shuffled-content drop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A: no communication | 10 | 84.5 [83-86] | 84.7 ± 1.8 | - | - | - | - | 82.2 | 0 | 0.0 | 41 | 64.8 | - |
| R1 | 10 | 87.0 [84-90] | 87.0 ± 2.6 | +2.5 | 0.636 | -0.8 | 1.000 | 87.0 | 0 | 1.0 | 41 | 66.1 | +19.0 |
| R2 | 10 | 84.0 [83-85] | 84.3 ± 2.5 | -0.5 | 1.000 | -3.8 | 0.615 | 81.0 | 0 | 1.7 | 42 | 64.0 | +24.2 |
| R4 | 10 | 87.5 [84-89] | 86.8 ± 2.2 | +3.0 | 0.615 | -0.2 | 1.000 | 84.0 | 0 | 2.3 | 43 | 65.6 | +21.8 |
| R8 | 10 | 88.0 [87-89] | 87.8 ± 1.8 | +3.5 | 0.123 | +0.2 | 1.000 | 82.8 | 0 | 2.4 | 42 | 67.6 | +20.5 |
| Rinf | 10 | 87.8 [85-90] | 87.0 ± 3.3 | +3.2 | 0.615 | - | - | 84.0 | 0 | 2.4 | 42 | 68.0 | +24.8 |

Trend over finite budgets (incl. 0): Spearman rho = +0.28, p = 0.0475 (n = 50 runs).
Saturation fit y(B) = y_inf - (y_inf - y0)·exp(-B/beta): y0 = 85.0, y_inf = 87.4, **B90 = 7.9** messages (95% seed-bootstrap CI 0.1-inf); smallest tested budget whose median reaches 90% of the gain: **4** (CI 1-inf).

## Other learned-sender variants

| Configuration | Seeds | Goals % median [IQR] | Δ vs L4 | p |
|---|---|---|---|---|
| L4 | 10 | 18.8 [12-27] | - | - |
| L4-team | 10 | 17.5 [11-23] | -1.2 | 0.811 |
| C | 10 | 13.5 [12-18] | -5.2 | 0.565 |
| Linf | 10 | 46.0 [40-54] | +27.2 | 0.000 |
