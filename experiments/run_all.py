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

from experiments.train import train, COMM_SETTINGS, FIXED_SENDERS
from env.rocket_env import MAX_STEPS
from utils.plot_results import summarize, plot
from utils.report import build_report

DEFAULT_SEEDS = tuple(range(10))

PARTIAL_RHO = 0.15   # partial observability level for the budget and delay sweeps

# (setting, obs_frac, comm_delay) triples per sweep
SWEEPS = {
    # RQ1: budget sweep 0 -> inf, full vs partial observability
    "budget": [(s, rho, 0) for rho in (1.0, PARTIAL_RHO) for s in COMM_SETTINGS],
    # RQ2: delay on unlimited communication (partial obs, where messages carry information)
    "delay": [("A", PARTIAL_RHO, 0)] + [("B", PARTIAL_RHO, d) for d in (0, 1, 2, 5)],
    # RQ3: value of communication as observability shrinks
    "obs": [(s, rho, 0) for rho in (1.0, 0.25, PARTIAL_RHO, 0.1) for s in ("A", "B")],
}
LEARNED = [s for s in COMM_SETTINGS if s not in FIXED_SENDERS]
SWEEPS["budget"] = [(s, rho, 0) for rho in (1.0, PARTIAL_RHO) for s in LEARNED]
# Fixed-schedule senders: the same three questions without a learned message head
SWEEPS["fixed"] = (
    [(s, PARTIAL_RHO, 0) for s in FIXED_SENDERS]                       # RQ1
    + [(s, 0.1, 0) for s in FIXED_SENDERS if s != "Finf"]              # RQ1 where sharing matters most
    + [("Finf", PARTIAL_RHO, d) for d in (1, 2, 5)]                     # RQ2
    + [("Finf", rho, 0) for rho in (1.0, 0.25, 0.1)]                    # RQ3
)
# Pilot of the event-triggered learned sender vs a fixed schedule (A is reused)
SWEEPS["pilot"] = [(s, PARTIAL_RHO, 0) for s in ("D5", "D", "B", "F20")]
# Quick end-to-end check of the pipeline
SWEEPS["smoke"] = [("A", 1.0, 0), ("B", 1.0, 0), ("A", PARTIAL_RHO, 0), ("D", PARTIAL_RHO, 0), ("B", PARTIAL_RHO, 0)]
SWEEPS["all"] = list(dict.fromkeys(SWEEPS["budget"] + SWEEPS["delay"] + SWEEPS["obs"] + SWEEPS["fixed"]))


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
    episodes: int = 4000,
    seeds=DEFAULT_SEEDS,
    max_steps: int = MAX_STEPS,
    save_dir: str = "checkpoints/sweep_v3",
    eval_every: int = 250,
    n_eval_episodes: int = 100,
    n_final_eval: int = 200,
    workers: int = 8,
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
                n_final_eval=n_final_eval,
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
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS)
    parser.add_argument("--save_dir", default="checkpoints/sweep_v3")
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
    )
