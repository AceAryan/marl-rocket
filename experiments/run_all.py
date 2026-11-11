"""
Run the communication sweeps in parallel and summarise the results.

Runs are stored as <save_dir>/obs<rho>/delay<d>/setting_<X>/seed_<s>/history.json.
A run whose history.json already exists is skipped, so an interrupted sweep
can be resumed by re-running the same command, and sweeps that share runs
(e.g. setting B at rho=0.25, delay 0) only train them once.

Usage:
    python -m experiments.run_all --sweep budget --workers 16
    python -m experiments.run_all --sweep all --episodes 3000 --seeds 0 1 2 3 4
"""

import sys, os
# Tiny matmuls: multithreaded BLAS only adds overhead (and oversubscribes parallel sweeps)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiments.train import train, FIXED_SENDERS, RULE_SENDERS
from env.rocket_env import MAX_STEPS
from utils.plot_results import summarize, plot
from utils.report import build_report

DEFAULT_SEEDS = tuple(range(10))

# Main observation radius for the budget and delay sweeps. With memory and
# 80-step episodes, no communication scores 92-96% at rho >= 0.10 (no headroom
# for communication); at 0.075 it falls to ~72% while unlimited communication
# reaches ~87%; at 0.05 nothing learns reliably in 10k episodes.
MAIN_RHO = 0.075
OBS_RHOS = (1.0, 0.15, 0.1, MAIN_RHO)   # RQ3: full view down to the main radius

LEARNED = ["A", "D2", "D5", "D10", "D", "C", "B"]   # A plus the learned-sender settings
UNLIMITED = ["B", "Rinf", "Finf"]                   # learned / rule / fixed sender, no budget

# (setting, obs_frac, comm_delay) triples per sweep
SWEEPS = {
    # RQ1: message budget, for each kind of sender
    "budget": [(s, MAIN_RHO, 0) for s in LEARNED],
    "rule": [(s, MAIN_RHO, 0) for s in RULE_SENDERS],
    "fixed": [(s, MAIN_RHO, 0) for s in FIXED_SENDERS],
    # RQ2: delay on unlimited communication
    "delay": [("A", MAIN_RHO, 0)] + [(s, MAIN_RHO, d) for s in UNLIMITED for d in (0, 2, 5)],
    # RQ3: value of communication as observability shrinks
    "obs": [(s, rho, 0) for rho in OBS_RHOS for s in ["A"] + UNLIMITED],
}
# Pilot: no communication vs the unlimited senders at the main radius
SWEEPS["pilot"] = [(s, MAIN_RHO, 0) for s in ["A"] + UNLIMITED]
# Quick end-to-end check of the pipeline
SWEEPS["smoke"] = [(s, MAIN_RHO, 0) for s in ("A", "D", "B", "R5", "F5")]
SWEEPS["all"] = list(dict.fromkeys(
    SWEEPS["budget"] + SWEEPS["rule"] + SWEEPS["fixed"] + SWEEPS["delay"] + SWEEPS["obs"]))


def run_dir(save_dir, setting, obs_frac, comm_delay, seed):
    return os.path.join(
        save_dir, f"obs{obs_frac}", f"delay{comm_delay}", f"setting_{setting}", f"seed_{seed}"
    )


def _run_one(job):
    try:
        t0 = time.time()
        h = train(verbose=False, **job)
        fe = h["final_eval"]
        return (
            f"done  {job['setting']:>3} obs={job['obs_frac']} d={job['comm_delay']} "
            f"seed={job['seed']} | R={fe['return_mean']:6.1f} goal={fe['goal_rate']:.2f} "
            f"concede={fe['concede_rate']:.2f} | {time.time() - t0:.0f}s"
        )
    except Exception:
        return f"FAILED {job}\n{traceback.format_exc()}"


def run_sweep(
    sweep: str = "budget",
    episodes: int = 10_000,
    seeds=DEFAULT_SEEDS,
    max_steps: int = MAX_STEPS,
    save_dir: str = "checkpoints/sweep_v4",
    eval_every: int = 250,
    n_eval_episodes: int = 100,
    n_final_eval: int = 200,
    workers: int = 8,
    msg_credit: str = "voi",
    msg_cost: float = 0.05,
    msg_handover: float = 0.5,
    memory: bool = True,
):
    jobs = []
    for setting, obs_frac, comm_delay in SWEEPS[sweep]:
        for seed in seeds:
            path = run_dir(save_dir, setting, obs_frac, comm_delay, seed)
            if os.path.exists(os.path.join(path, "history.json")):
                continue
            jobs.append(dict(
                setting=setting, obs_frac=obs_frac, comm_delay=comm_delay, seed=seed,
                n_episodes=episodes, max_steps=max_steps, save_dir=path,
                eval_every=eval_every, n_eval_episodes=n_eval_episodes,
                n_final_eval=n_final_eval, msg_credit=msg_credit, msg_cost=msg_cost,
                msg_handover=msg_handover, memory=memory,
            ))
    total = len(SWEEPS[sweep]) * len(seeds)
    print(f"Sweep '{sweep}': {total} runs, {total - len(jobs)} already done, "
          f"{len(jobs)} to train on {workers} workers", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_one, job) for job in jobs]
        for k, fut in enumerate(as_completed(futures), 1):
            print(f"[{k}/{len(jobs)} | {(time.time() - t0) / 60:.0f} min] {fut.result()}", flush=True)

    summarize(save_dir)
    plot(save_dir, save=True)
    build_report(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default="budget", choices=list(SWEEPS))
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--save_dir", default="checkpoints/sweep_v4")
    parser.add_argument("--msg_credit", default="voi", choices=["voi", "team"])
    parser.add_argument("--msg_cost", type=float, default=0.05)
    parser.add_argument("--msg_handover", type=float, default=0.5)
    parser.add_argument("--no_memory", action="store_true", help="ablation: blank the memory inputs")
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument("--n_eval_episodes", type=int, default=100)
    parser.add_argument("--n_final_eval", type=int, default=200)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    args = parser.parse_args()
    run_sweep(
        sweep=args.sweep,
        episodes=args.episodes,
        seeds=args.seeds,
        max_steps=args.max_steps,
        save_dir=args.save_dir,
        eval_every=args.eval_every,
        n_eval_episodes=args.n_eval_episodes,
        n_final_eval=args.n_final_eval,
        workers=args.workers,
        msg_credit=args.msg_credit,
        msg_cost=args.msg_cost,
        msg_handover=args.msg_handover,
        memory=not args.no_memory,
    )
