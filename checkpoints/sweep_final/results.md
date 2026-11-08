# Results

Final greedy evaluation on 200 held-out episodes per seed. Goal rate = share of episodes with an assisted goal. Δ and p: Welch t-test against A at the same observation radius (* p<0.05, ** p<0.01). Steps to goal = mean episode length (200 = no goal). *Goal % silenced* = the same policy evaluated with the channel muted (how much it relies on messages).

## RQ1: message budget (rho = 0.15)

| Setting | Seeds | Goal rate % (95% CI) | Δ vs A (pp) | p | Steps to goal | Msgs/agent | Goal % silenced |
|---|---|---|---|---|---|---|---|
| A: no communication | 5 | 90.6 ± 10.3 | - | - | 67 | 0.0 | - |
| D5: learned, budget 5 | 2 | 69.8 ± 123.9 | -20.8 | 0.247 | 114 | 4.5 | 45.8 |
| D: learned, budget 20 | 2 | 79.5 ± 12.7 | -11.1 * | 0.038 | 108 | 18.9 | 35.8 |
| B: learned, unlimited | 2 | 95.0 ± 12.7 | +4.4 | 0.308 | 58 | 25.3 | 65.5 |
| F20: fixed, every 10 steps | 2 | 86.0 ± 108.0 | -4.6 | 0.686 | 84 | 8.0 | 6.2 |

## RQ1: message budget (full view, rho = 1.0)

| Setting | Seeds | Goal rate % (95% CI) | Δ vs A (pp) | p | Steps to goal | Msgs/agent | Goal % silenced |
|---|---|---|---|---|---|---|---|
| A: no communication | 5 | 97.9 ± 1.4 | - | - | 50 | 0.0 | - |

## RQ2: message delay (rho = 0.15)

| Setting | Seeds | Goal rate % (95% CI) | Δ vs A (pp) | p | Steps to goal | Msgs/agent | Goal % silenced |
|---|---|---|---|---|---|---|---|
| A: no communication | 5 | 90.6 ± 10.3 | - | - | 67 | 0.0 | - |
| B: learned, unlimited | 2 | 95.0 ± 12.7 | +4.4 | 0.308 | 58 | 25.3 | 65.5 |

## RQ3: observability

| rho | A: no communication | B: learned, unlimited | Finf: fixed, every step |
|---|---|---|---|
| 1 | 97.9 ± 1.4 | - | - |
| 0.25 | 96.1 ± 4.5 | - | - |
| 0.15 | 90.6 ± 10.3 | 95.0 ± 12.7 | - |
| 0.1 | 78.2 ± 4.3 | - | - |

## Message timing (greedy, rho = 0.15)

| Policy | Seeds | Median send step | % sent in first 10 steps | % sent while seeing ball |
|---|---|---|---|---|
| Learned D, before fix | 5 | 12 | 40 | 60 |
| Learned D, event-triggered | 2 | 26 | 3 | 100 |
| Fixed F20 | 2 | 49 | 11 | 68 |
