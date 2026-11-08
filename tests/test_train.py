"""Tests for the training-side message logic and an end-to-end train() smoke."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tempfile
import numpy as np

from env.rocket_env import N_ACTIONS, N_MESSAGES, OBS_DIM
from agents.dqn_agent import DQNAgent
from experiments.train import (
    BALL_VIS, FIXED_SENDERS, choose_messages, decision_points, make_env, train,
)


def _agents():
    return [DQNAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS, n_messages=N_MESSAGES, agent_id=i)
            for i in range(2)]


def _obs(ball_visible):
    o = np.zeros(OBS_DIM, np.float32)
    o[BALL_VIS] = 1.0 if ball_visible else 0.0
    return o


def test_learned_sender_silent_without_ball_sighting():
    agents = _agents()
    for a in agents:
        a.eps = 1.0   # fully random: would send often if allowed
    for _ in range(200):
        msgs, decided = choose_messages(agents, [_obs(False), _obs(False)], [True, True], fixed=False)
        assert msgs == [0, 0] and decided == [False, False]


def test_learned_sender_silent_when_budget_spent():
    agents = _agents()
    for a in agents:
        a.eps = 1.0
    for _ in range(200):
        msgs, decided = choose_messages(agents, [_obs(True), _obs(True)], [False, False], fixed=False)
        assert msgs == [0, 0] and decided == [False, False]


def test_decision_points_require_sighting_and_permission():
    obs = [_obs(True), _obs(False)]
    assert decision_points(obs, [True, True]) == [True, False]
    assert decision_points(obs, [False, True]) == [False, False]


def test_silence_blocks_all_messages():
    agents = _agents()
    for a in agents:
        a.eps = 1.0
    for fixed in (False, True):
        msgs, _ = choose_messages(agents, [_obs(True), _obs(True)], [True, True], fixed, silence=True)
        assert msgs == [0, 0]


def test_exploration_is_silence_biased():
    agent = _agents()[0]
    agent.eps = 1.0
    picks = [agent.select_message(_obs(True)) for _ in range(4000)]
    silent = np.mean(np.array(picks) == 0)
    assert 0.45 < silent < 0.55
    assert set(picks) == set(range(N_MESSAGES))


def test_fixed_sender_sends_every_k_steps():
    for setting, k in FIXED_SENDERS.items():
        env = make_env(setting, obs_frac=0.15, max_steps=200)
        env.assist_required = False
        obs, info = env.reset(seed=0)
        agents = _agents()
        sent_steps = []
        for t in range(60):
            msgs, _ = choose_messages(agents, obs, info["can_send"], fixed=True)
            if msgs[0]:
                sent_steps.append(t)
            obs, _, term, trunc, info = env.step([0, 0], msgs)
            if term or trunc:
                break
        assert sent_steps and sent_steps[0] < k, setting
        assert all(b - a == k for a, b in zip(sent_steps, sent_steps[1:])), setting


def test_train_smoke_all_kinds():
    with tempfile.TemporaryDirectory() as d:
        for setting in ("A", "D", "B", "F20"):
            h = train(setting=setting, n_episodes=3, obs_frac=0.15, max_steps=40,
                      save_dir=os.path.join(d, setting), eval_every=3, n_eval_episodes=2,
                      n_final_eval=2, verbose=False, seed=0)
            assert "final_eval" in h and len(h["eval"]) == 1
            assert h["fixed_sender"] == (setting == "F20")
            assert h["event_triggered"] == (setting in ("D", "B"))
            assert ("final_eval_silenced" in h) == (setting != "A")
            assert os.path.exists(os.path.join(d, setting, "history.json"))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("All tests passed!")
