"""Tests for the training-side message logic and an end-to-end train() smoke."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tempfile
import numpy as np

from env.rocket_env import N_ACTIONS, N_MESSAGES, OBS_DIM, OBS_SLICES, RocketEnv
from agents.dqn_agent import DQNAgent
from experiments.train import (
    BALL_VIS, FIXED_SENDERS, NO_MSG, choose_messages, decision_points, make_env, mask_message,
    message_value, train,
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


def test_mask_message_matches_env_no_message_encoding():
    env = RocketEnv(obs_frac=0.25, opponent=False)
    obs, _ = env.reset(seed=0)
    silent = obs[1].copy()                       # holds no message yet
    obs, *_ = env.step([0, 0], [3, 0])           # agent 0 messages agent 1
    assert obs[1][NO_MSG] == 0.0
    masked = mask_message(obs[1])
    for key in ("msg_token", "msg_ball", "msg_ball_vis", "msg_age", "msg_sender"):
        assert np.array_equal(masked[OBS_SLICES[key]], silent[OBS_SLICES[key]]), key
    others = [i for i in range(OBS_DIM) if not any(
        OBS_SLICES[k].start <= i < OBS_SLICES[k].stop
        for k in ("msg_token", "msg_ball", "msg_ball_vis", "msg_age", "msg_sender"))]
    assert np.array_equal(masked[others], obs[1][others])


def test_message_value():
    agent = _agents()[0]
    o = np.zeros(OBS_DIM, np.float32)
    o[NO_MSG] = 1.0
    o[OBS_SLICES["msg_age"]] = 1.0
    assert message_value(agent, o) == 0.0              # no message held
    o_msg = o.copy()
    o_msg[NO_MSG] = 0.0
    o_msg[OBS_SLICES["msg_token"].start + 2] = 1.0
    o_msg[OBS_SLICES["msg_ball"]] = [0.5, -0.5, 0.0, 0.0]
    o_msg[OBS_SLICES["msg_ball_vis"]] = 1.0
    o_msg[OBS_SLICES["msg_age"]] = 0.0
    q = agent.q_net(o_msg)
    a_cf = int(np.argmax(agent.q_net(o)))
    v = message_value(agent, o_msg)
    assert v >= 0.0 and np.isclose(v, q.max() - q[a_cf])


def test_send_age_observed():
    from env.rocket_env import MSG_TTL
    env = RocketEnv()
    obs, _ = env.reset(seed=0)
    assert obs[0][OBS_SLICES["send_age"]][0] == 1.0     # never sent
    obs, *_ = env.step([0, 0], [1, 0])
    assert obs[0][OBS_SLICES["send_age"]][0] == 0.0 and obs[1][OBS_SLICES["send_age"]][0] == 1.0
    for _ in range(3):
        obs, *_ = env.step([0, 0], [0, 0])
    assert np.isclose(obs[0][OBS_SLICES["send_age"]][0], 3 / MSG_TTL)


def test_train_smoke_all_kinds():
    with tempfile.TemporaryDirectory() as d:
        for setting in ("A", "D", "B", "F20"):
            h = train(setting=setting, n_episodes=3, obs_frac=0.15, max_steps=40,
                      save_dir=os.path.join(d, setting), eval_every=3, n_eval_episodes=2,
                      n_final_eval=2, verbose=False, seed=0)
            assert "final_eval" in h and "final_eval_last" in h and len(h["eval"]) == 1
            assert h["best_episode"] in h["eval_episode"]
            assert h["fixed_sender"] == (setting == "F20")
            assert h["event_triggered"] == (setting in ("D", "B"))
            assert ("final_eval_silenced" in h) == (setting != "A")
            assert os.path.exists(os.path.join(d, setting, "history.json"))
        h = train(setting="D5", n_episodes=3, obs_frac=0.15, max_steps=40, msg_credit="team",
                  save_dir=os.path.join(d, "team"), eval_every=3, n_eval_episodes=2,
                  n_final_eval=2, verbose=False, seed=0)
        assert h["msg_credit"] == "team" and h["msg_cost"] is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("All tests passed!")
