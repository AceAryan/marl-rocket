"""
Experiment configuration shared by run_all, the report and the plots.

A job is a dict of train() keyword arguments (without seed/save_dir). Sweeps
are lists of jobs; run_all expands them over seeds.

The main-sweep regime (MAIN_MODE, MAIN_RHO, EPISODES, MSG_CREDIT, SHARE,
LR_DECAY) is set by the pre-registered pilot rules (README, "Pilot"):
`python -m utils.report --dir <pilot dir> --pilot --apply` writes
experiments/pilot_decisions.json, which overrides the defaults below. Do not
edit the regime by hand.
"""

import json
import os

# ---------------------------------------------------------------- main regime (defaults; the pilot overrides)
MAIN_MODE = "single"          # "single" (one goal ends the episode) or "match" (timed multi-goal)
MAIN_RHO = 0.075              # observation radius of the budget and delay sweeps
EPISODES = {"single": 10_000, "match": 4_000}
MAX_STEPS = {"single": 80, "match": 200}
EVAL = {"single": dict(eval_every=250, n_eval_episodes=100, n_final_eval=200),
        "match": dict(eval_every=200, n_eval_episodes=50, n_final_eval=100)}
MSG_CREDIT = "rollout"
SHARE = True
LR_DECAY = False

DECISIONS_FILE = os.path.join(os.path.dirname(__file__), "pilot_decisions.json")
PILOT_DECISIONS = None


def apply_decisions(decisions):
    """Set the main regime from the pilot's decisions (utils.report --pilot)."""
    global PILOT_DECISIONS, MAIN_MODE, MAIN_RHO, SHARE, MSG_CREDIT, LR_DECAY, EPISODES
    PILOT_DECISIONS = dict(decisions)
    MAIN_MODE = decisions["mode"]
    MAIN_RHO = float(decisions["rho"])
    SHARE = bool(decisions["share"])
    MSG_CREDIT = decisions["credit"]
    LR_DECAY = bool(decisions["lr_decay"])
    EPISODES = dict(EPISODES, **{MAIN_MODE: int(decisions["episodes"])})


if os.path.exists(DECISIONS_FILE):
    with open(DECISIONS_FILE) as _f:
        apply_decisions(json.load(_f))

# Message budgets per agent per episode (single-goal mode); match mode scales them.
# Fixed in advance: they span the rule sender's usage (~2-3 per episode, it sends at
# most once per message lifetime), the learned gate's (~5) and the fixed schedule's
# (up to every step).
BUDGETS = (1, 2, 3, 4, 6, 8, 12, 16, 24)
FIXED_BUDGETS = (2, 4, 8, 16, 32)
BLIND_BUDGETS = (2, 4, 8)                # budget-awareness ablation (feedback 2)
SECOND_BUDGETS = (1, 2, 4, 8)            # rule sender at SECOND_RHO
DELAYS = (1, 2, 4, 8)
RHOS = (1.0, 0.25, 0.15, 0.1, 0.075, 0.06)


def second_rho():
    """Second radius for the saturation curve (RQ3)."""
    return 0.1 if MAIN_RHO != 0.1 else 0.075


def obs_rhos():
    """RQ3 radii besides MAIN_RHO: every wider view, plus the second radius."""
    return tuple(r for r in RHOS if (r > MAIN_RHO or r == second_rho()) and r != MAIN_RHO)
MATCH_BUDGET_SCALE = 2.5                 # 200-step matches vs 80-step episodes

MAIN_SEEDS = tuple(range(10))
PILOT_SEEDS = (100, 101, 102)


def _b(budget, mode):
    """A single-goal budget expressed for the given mode."""
    if budget is None:
        return "inf"
    return str(budget if mode == "single" else max(1, round(budget * MATCH_BUDGET_SCALE)))


def job(setting, rho=None, delay=0, mode=None, **kw):
    mode = mode or MAIN_MODE
    j = dict(setting=setting, obs_frac=MAIN_RHO if rho is None else rho, comm_delay=delay,
             match=(mode == "match"), max_steps=MAX_STEPS[mode], n_episodes=EPISODES[mode],
             share=SHARE, msg_credit=MSG_CREDIT, lr_decay=LR_DECAY, budget_obs=True)
    j.update(EVAL[mode])
    j.update(kw)
    return j


def _grid(prefix, budgets, mode=None, **kw):
    mode = mode or MAIN_MODE
    return [job(f"{prefix}{_b(b, mode)}", mode=mode, **kw) for b in list(budgets) + [None]]


def main_sweeps():
    m = MAIN_MODE
    tier1 = [job("A")] + _grid("L", BUDGETS) + _grid("R", BUDGETS) + _grid("F", FIXED_BUDGETS)
    tier2 = ([job(f"L{_b(b, m)}", budget_obs=False) for b in BLIND_BUDGETS]
             + [job(s, delay=d) for s in ("Linf", "Rinf") for d in DELAYS])
    tier3 = [job(s, rho=r) for r in obs_rhos() for s in ("A", "Linf", "Rinf") if not (r >= 1.0 and s == "Rinf")]
    tier3 += [job(f"R{_b(b, m)}", rho=second_rho()) for b in SECOND_BUDGETS]
    other = "team" if MSG_CREDIT == "rollout" else "rollout"
    tier4 = [job("C"), job(f"L{_b(4, m)}", msg_credit=other)]
    return {"tier1": tier1, "tier2": tier2, "tier3": tier3, "tier4": tier4}


def pilot_sweeps():
    """Pre-registered pilot (README, "Pilot"): sharing check, regime, senders."""
    single = dict(mode="single", n_episodes=15_000)
    match = dict(mode="match", n_episodes=5_000)
    p1 = [job(s, rho=0.075, share=sh, **single) for s in ("A", "Finf") for sh in (True, False)]
    p2 = [job(s, rho=r, **kw) for kw in (single, match) for r in (0.06, 0.075, 0.1) for s in ("A", "Finf")]
    p3 = []
    for kw, b in ((single, 4), (match, 10)):
        p3 += [job(f"R{b}", rho=0.075, **kw), job("Rinf", rho=0.075, **kw), job(f"L{b}", rho=0.075, **kw),
               job(f"L{b}", rho=0.075, msg_credit="team", **kw), job("Linf", rho=0.075, **kw)]
    return {"pilot_share": p1, "pilot_regime": p2, "pilot_senders": p3}


def smoke_sweep():
    kw = dict(mode="single", n_episodes=30, eval_every=10, n_eval_episodes=3, n_final_eval=4)
    return [job("A", **kw), job("L2", **kw), job("R2", **kw), job("Finf", **kw),
            job("L4", mode="match", n_episodes=6, max_steps=60, eval_every=3, n_eval_episodes=2, n_final_eval=2)]


def sweeps():
    s = dict(main_sweeps())
    s["main"] = [j for t in ("tier1", "tier2", "tier3", "tier4") for j in s[t]]
    s.update(pilot_sweeps())
    s["pilot"] = [j for t in ("pilot_share", "pilot_regime", "pilot_senders") for j in s[t]]
    s["smoke"] = smoke_sweep()
    return s
