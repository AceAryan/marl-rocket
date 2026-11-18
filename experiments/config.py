"""
Experiment configuration shared by run_all, the report and the plots.

A job is a dict of train() keyword arguments (without seed/save_dir). Sweeps
are lists of jobs; run_all expands them over seeds.

The main-sweep regime (MAIN_MODE, MAIN_RHO, EPISODES, BUDGETS, MSG_CREDIT,
SHARE, LR_DECAY) is fixed by the pre-registered pilot rules (README,
"Pilot"); edit it only through that procedure.
"""

# ---------------------------------------------------------------- main regime (set by the pilot)
MAIN_MODE = "single"          # "single" (one goal ends the episode) or "match" (timed multi-goal)
MAIN_RHO = 0.075              # observation radius of the budget and delay sweeps
SECOND_RHO = 0.1              # second radius for the saturation curve (RQ3)
EPISODES = {"single": 10_000, "match": 4_000}
MAX_STEPS = {"single": 80, "match": 200}
EVAL = {"single": dict(eval_every=250, n_eval_episodes=100, n_final_eval=200),
        "match": dict(eval_every=200, n_eval_episodes=50, n_final_eval=100)}
MSG_CREDIT = "rollout"
SHARE = True
LR_DECAY = False

# Message budgets per agent per episode (single-goal mode); match mode scales them.
BUDGETS = (1, 2, 3, 4, 6, 8, 12, 16, 24)
FIXED_BUDGETS = (2, 4, 8, 16)
BLIND_BUDGETS = (2, 4, 8)                # budget-awareness ablation (feedback 2)
SECOND_BUDGETS = (1, 2, 4, 8)            # rule sender at SECOND_RHO
DELAYS = (1, 2, 4, 8)
OBS_RHOS = (1.0, 0.25, 0.15, 0.1)        # RQ3, besides MAIN_RHO
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
    tier3 = [job(s, rho=r) for r in OBS_RHOS for s in ("A", "Linf", "Rinf") if not (r >= 1.0 and s == "Rinf")]
    tier3 += [job(f"R{_b(b, m)}", rho=SECOND_RHO) for b in SECOND_BUDGETS if SECOND_RHO not in (MAIN_RHO,)]
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
