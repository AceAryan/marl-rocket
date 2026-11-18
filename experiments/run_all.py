"""
Run sweeps in parallel and summarise the results.

Runs are stored as <save_dir>/<mode>/obs<rho>/delay<d>/<label>/seed_<s>/history.json
(mode = single | match; label = setting plus any variant, e.g. L4-blind).
A run whose history.json is complete and was made with the same configuration
is skipped, so an interrupted sweep is resumed by re-running the command; a
history made with a different configuration stops the sweep (use a new
--save_dir instead of mixing configurations).

Robustness: jobs are queued seed-major (a partial sweep stays balanced across
configurations); a crashed worker pool is rebuilt and unfinished jobs are
resubmitted; on Windows the PC is kept awake while the sweep runs; progress is
appended to <save_dir>/sweep_log.txt; the summary and report are written even
if the sweep stops early.

Usage:
    python -m experiments.run_all --sweep pilot --save_dir checkpoints/pilot_v6 --workers 20
    python -m experiments.run_all --sweep tier1 --workers 20      # main sweep, tier by tier
    python -m experiments.run_all --sweep smoke --save_dir checkpoints/smoke --workers 4
"""

import sys, os
# Tiny matmuls: multithreaded BLAS only adds overhead (and oversubscribes parallel sweeps)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

from experiments import config as C
from experiments.train import train, run_label

DEFAULT_SAVE_DIR = "checkpoints/sweep_v5"
MAX_POOL_RESTARTS = 3
# Fields of history["config"] that must match for a finished run to be reused
CHECKED_KEYS = ("setting", "n_episodes", "obs_frac", "comm_delay", "max_steps", "match", "share",
                "msg_credit", "msg_cost", "msg_handover", "budget_obs", "memory", "lr_decay",
                "eval_every", "n_eval_episodes", "n_final_eval", "seed")


def run_dir(save_dir, job, seed):
    mode = "match" if job["match"] else "single"
    label = run_label(job["setting"], job["budget_obs"], job["msg_credit"], job["share"], job["lr_decay"])
    return os.path.join(save_dir, mode, f"obs{job['obs_frac']}", f"delay{job['comm_delay']}", label, f"seed_{seed}")


def finished(path, job):
    """True if a complete history with a matching configuration exists; raises
    if one exists with a different configuration; False otherwise."""
    p = os.path.join(path, "history.json")
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            h = json.load(f)
    except (OSError, ValueError):
        return False   # unreadable or truncated: train again
    if "final_eval" not in h:
        return False
    cfg = h.get("config", {})
    diff = {k: (cfg.get(k), job[k]) for k in CHECKED_KEYS if k in job and cfg.get(k) != job[k]}
    if diff:
        raise RuntimeError(f"{p} was made with a different configuration {diff}; use another --save_dir")
    return True


def _run_one(job):
    marker = os.environ.get("RUN_ALL_CRASH_ONCE")   # tests: kill one worker to exercise pool recovery
    if marker and not os.path.exists(marker):
        open(marker, "w").close()
        os._exit(1)
    try:
        t0 = time.time()
        h = train(verbose=False, **job)
        fe, fl = h["final_eval"], h["final_eval_last"]
        return (
            f"done  {h['label']:>10} {'match' if job['match'] else 'single'} obs={job['obs_frac']} "
            f"d={job['comm_delay']} seed={job['seed']} | goals={fe['goals_mean']:.2f} (last {fl['goals_mean']:.2f}) "
            f"msgs={fe['msgs_mean'] / 2:.1f} | {time.time() - t0:.0f}s"
        )
    except Exception:
        return f"FAILED {job}\n{traceback.format_exc()}"


class _KeepAwake:
    """Windows: keep the system from sleeping while a sweep runs."""
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001

    def __enter__(self):
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED)
        return self

    def __exit__(self, *exc):
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
        return False


def build_jobs(sweep, seeds, save_dir, episodes=None):
    specs = C.sweeps()[sweep]
    jobs, seen = [], set()
    for seed in seeds:                       # seed-major: balanced if the sweep stops early
        for spec in specs:
            j = dict(spec, seed=seed)
            if episodes is not None:
                j["n_episodes"] = episodes
            path = run_dir(save_dir, j, seed)
            if path in seen:
                continue
            seen.add(path)
            j["save_dir"] = path
            jobs.append(j)
    return jobs


def run_sweep(sweep="tier1", seeds=None, save_dir=DEFAULT_SAVE_DIR, workers=8, episodes=None,
              report=True, dry_run=False):
    seeds = seeds if seeds is not None else (C.PILOT_SEEDS if sweep.startswith("pilot") else C.MAIN_SEEDS)
    all_jobs = build_jobs(sweep, seeds, save_dir, episodes)
    todo = [j for j in all_jobs if not finished(j["save_dir"], j)]
    log_path = os.path.join(save_dir, "sweep_log.txt")

    def log(msg):
        print(msg, flush=True)
        if not dry_run:
            os.makedirs(save_dir, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")

    log(f"Sweep '{sweep}' {time.strftime('%Y-%m-%d %H:%M')}: {len(all_jobs)} runs, "
        f"{len(all_jobs) - len(todo)} already done, {len(todo)} to train on {workers} workers")
    if dry_run:
        for j in todo:
            log("  " + j["save_dir"])
        return
    t0 = time.time()
    try:
        with _KeepAwake():
            for attempt in range(MAX_POOL_RESTARTS + 1):
                pending = [j for j in todo if not finished(j["save_dir"], j)]
                if not pending:
                    break
                if attempt:
                    log(f"Rebuilding the worker pool (attempt {attempt}); {len(pending)} runs left")
                done = 0
                try:
                    with ProcessPoolExecutor(max_workers=workers) as pool:
                        futures = [pool.submit(_run_one, j) for j in pending]
                        for fut in as_completed(futures):
                            done += 1
                            log(f"[{done}/{len(pending)} | {(time.time() - t0) / 60:.0f} min] {fut.result()}")
                    break
                except BrokenProcessPool:
                    log(f"Worker pool broke after {done} runs (a worker died, e.g. the PC slept or ran out of memory)")
            else:
                log("Giving up after repeated pool failures; re-run the same command to resume")
    finally:
        if report:
            try:
                from utils.report import build_report
                build_report(save_dir)
            except Exception:
                log("Report failed:\n" + traceback.format_exc())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default="tier1", choices=sorted(C.sweeps()))
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--save_dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--episodes", type=int, default=None, help="override the episode count (smoke tests)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--no_report", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="list the runs that would be trained")
    args = parser.parse_args()
    run_sweep(sweep=args.sweep, seeds=args.seeds, save_dir=args.save_dir, workers=args.workers,
              episodes=args.episodes, report=not args.no_report, dry_run=args.dry_run)
