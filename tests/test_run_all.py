"""Sweep runner: resume, configuration checks and recovery from a crashed worker."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import tempfile

from experiments import config as C
from experiments import run_all


def _tiny(monkeypatch_target):
    kw = dict(mode="single", n_episodes=3, eval_every=3, n_eval_episodes=1, n_final_eval=1, max_steps=20)
    return [C.job("A", **kw), C.job("R2", **kw)]


def test_pool_recovery_resume_and_config_check():
    orig = C.sweeps
    try:
        C.sweeps = lambda: {"tiny": _tiny(None)}
        with tempfile.TemporaryDirectory() as d:
            os.environ["RUN_ALL_CRASH_ONCE"] = os.path.join(d, "crashed")
            run_all.run_sweep("tiny", seeds=[0, 1], save_dir=d, workers=2, report=False)
            assert os.path.exists(os.path.join(d, "crashed"))
            jobs = run_all.build_jobs("tiny", [0, 1], d)
            assert len(jobs) == 4 and all(run_all.finished(j["save_dir"], j) for j in jobs)
            log = open(os.path.join(d, "sweep_log.txt"), encoding="utf-8").read()
            assert "Rebuilding the worker pool" in log
            # A finished run made with another configuration is never silently reused
            j = dict(jobs[0], n_episodes=5)
            try:
                run_all.finished(j["save_dir"], j)
                raise AssertionError("config mismatch not detected")
            except RuntimeError:
                pass
            # A truncated history is retrained, not trusted
            with open(os.path.join(jobs[1]["save_dir"], "history.json"), "w") as f:
                f.write('{"setting": ')
            assert not run_all.finished(jobs[1]["save_dir"], jobs[1])
    finally:
        C.sweeps = orig
        os.environ.pop("RUN_ALL_CRASH_ONCE", None)


def test_jobs_are_seed_major_and_deduplicated():
    jobs = run_all.build_jobs("pilot", [100, 101], "x")
    seeds = [j["seed"] for j in jobs]
    assert seeds == sorted(seeds)                      # all of seed 100 before seed 101
    assert len({j["save_dir"] for j in jobs}) == len(jobs)
