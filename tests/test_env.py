"""Quick smoke tests. Run with `python tests/test_env.py` or `pytest tests`."""
import sys, os
import json
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from env.rocket_env import (
    BALL_RADIUS,
    FIELD_H,
    FIELD_W,
    GOAL_REWARD,
    N_ACTIONS,
    N_MESSAGES,
    OBS_DIM,
    OBS_SLICES,
    SHAPING_GAMMA,
    RocketEnv,
)
from agents.dqn_agent import DQNAgent, NStepAccumulator
from experiments.train import evaluate, train


def _freeze_ball(env, pos, vel=(0.0, 0.0)):
    env._ball_pos = np.array(pos, dtype=np.float32)
    env._ball_vel = np.array(vel, dtype=np.float32)


def _park_players(env):
    """Move everyone far from the ball's path so they don't touch it."""
    env._agent_pos[:] = [[2.0, 2.0], [FIELD_W - 2.0, 2.0]][: env.n_agents]
    env._opp_pos[:] = [2.0, FIELD_H - 2.0]


def test_reset_and_step_shapes():
    env = RocketEnv()
    obs_list, info = env.reset(seed=0)
    assert len(obs_list) == 2 and obs_list[0].shape == (OBS_DIM,)
    assert info["can_send"] == [True, True]
    next_obs, reward, term, trunc, info = env.step([0, 0], [1, 2])
    assert len(next_obs) == 2 and isinstance(reward, float)
    print("ok  reset/step shapes")


def test_observation_space_bounds():
    for obs_frac, budget in ((1.0, None), (0.25, 5)):
        env = RocketEnv(obs_frac=obs_frac, comm_budget=budget)
        obs_list, _ = env.reset(seed=1)
        rng = np.random.default_rng(0)
        for _ in range(600):
            assert all(env.observation_space.contains(o) for o in obs_list)
            obs_list, _, te, tr, _ = env.step(
                rng.integers(N_ACTIONS, size=2).tolist(), rng.integers(N_MESSAGES, size=2).tolist()
            )
            if te or tr:
                obs_list, _ = env.reset()
    print("ok  observations satisfy their declared space")


def test_team_goal():
    env = RocketEnv(assist_required=False)
    env.reset(seed=0)
    _park_players(env)
    _freeze_ball(env, [FIELD_W / 2, FIELD_H - 1.0], [0.0, 1.0])
    _, reward, terminated, truncated, info = env.step([0, 0])
    assert terminated and not truncated
    assert info["score_team"] == 1 and info["task_reward"] == GOAL_REWARD
    print("ok  team goal")


def _shot_after_touches(env, touches):
    """Agent 0 shoots from just below the goal mouth after the given touches."""
    env.reset(seed=0)
    _park_players(env)
    env._step_count = 50
    for who, step in touches:
        if who == "opp":
            env._last_opp_touch = step
        else:
            env._last_touch[who] = step
    env._agent_pos[0] = [FIELD_W / 2, FIELD_H - 2.3]
    _freeze_ball(env, [FIELD_W / 2, FIELD_H - 1.3])
    _, _, terminated, _, info = env.step([1, 0])   # agent 0 pushes the ball in
    for _ in range(3):
        if terminated:
            break
        _, _, terminated, _, info = env.step([0, 0])
    return terminated, info


def test_assisted_goal_counts():
    terminated, info = _shot_after_touches(RocketEnv(), [(1, 30)])
    assert terminated and info["score_team"] == 1
    print("ok  assisted goal counts")


def test_unassisted_goal_bounces():
    env = RocketEnv()
    for touches in ([], [(1, 30), ("opp", 40)], [(1, 50 - 41)]):  # none / opp since / too old
        terminated, info = _shot_after_touches(env, touches)
        assert not terminated and info["score_team"] == 0, touches
        assert env._ball_pos[1] < FIELD_H
    print("ok  unassisted goals bounce off the goal line")


def test_assist_shaping_pays_on_pass():
    from env.rocket_env import ASSIST_POTENTIAL, ASSIST_GAMMA
    env = RocketEnv(opponent=False)
    env.reset(seed=0)
    env._agent_pos[:] = [[10.0, 10.0], [10.0, 13.0]]
    _freeze_ball(env, [10.0, 11.0])
    _, _, _, _, info = env.step([1, 0])   # agent 0 touches: no live assist yet
    base_phi = env._potential()
    env._agent_pos[:] = [[2.0, 2.0], [env._ball_pos[0], env._ball_pos[1] + 0.6]]
    env._ball_vel[:] = 0.0
    phi_before = env._potential()
    _, _, _, _, info = env.step([0, 2])   # agent 1 touches: a goal by agent 1 would now count
    assert env._assisted(env._last_toucher)
    expected = env._potential() - phi_before + ASSIST_GAMMA * ASSIST_POTENTIAL
    assert np.isclose(info["shaping_reward"], expected), (info["shaping_reward"], expected)
    print("ok  the pass that creates a live assist pays the assist potential")


def test_assist_flags_observed():
    env = RocketEnv(opponent=False)
    env.reset(seed=0)
    env._agent_pos[:] = [[10.0, 10.0], [3.0, 3.0]]
    _freeze_ball(env, [10.0, 11.0])
    obs, *_ = env.step([1, 0])   # agent 0 touches the ball
    assert obs[0][OBS_SLICES["assist_mate"]][0] == 1.0 and obs[0][OBS_SLICES["assist_me"]][0] == 0.0
    assert obs[1][OBS_SLICES["assist_me"]][0] == 1.0 and obs[1][OBS_SLICES["assist_mate"]][0] == 0.0
    print("ok  assist eligibility is observable")


def test_opponent_goal():
    env = RocketEnv()
    env.reset(seed=0)
    _park_players(env)
    _freeze_ball(env, [FIELD_W / 2, 1.0], [0.0, -1.0])
    _, _, terminated, _, info = env.step([0, 0])
    assert terminated and info["score_opp"] == 1 and info["task_reward"] == -GOAL_REWARD
    print("ok  opponent goal")


def test_ball_bounces_outside_goal_mouth():
    env = RocketEnv()
    env.reset(seed=0)
    _park_players(env)
    _freeze_ball(env, [1.0, FIELD_H - 1.0], [0.0, 1.0])  # top-left corner, outside goal
    _, _, terminated, _, _ = env.step([0, 0])
    assert not terminated
    assert env._ball_vel[1] < 0 and env._ball_pos[1] <= FIELD_H - BALL_RADIUS
    print("ok  ball bounces outside the goal mouth")


def test_kick_direction():
    env = RocketEnv(opponent=False)
    env.reset(seed=0)
    env._agent_pos[:] = [[10.0, 10.0], [2.0, 2.0]]
    _freeze_ball(env, [10.0, 11.2])
    env.step([5, 0])  # boost up into the ball
    assert env._ball_vel[1] > 0.9 and abs(env._ball_vel[0]) < 1e-3
    print("ok  boosting into the ball kicks it forward")


def test_potential_reward_shaping():
    env = RocketEnv(opponent=False)
    env.reset(seed=0)
    _park_players(env)
    _freeze_ball(env, [10.0, 15.0])
    phi_before = env._potential()
    _, reward, _, _, info = env.step([0, 0])
    expected = SHAPING_GAMMA * env._potential() - phi_before
    assert info["task_reward"] == 0.0
    assert np.isclose(info["shaping_reward"], expected) and np.isclose(reward, expected)
    print("ok  potential-based reward shaping")


def test_comm_budget():
    env = RocketEnv(comm_budget=0)
    _, info = env.reset(seed=0)
    assert info["can_send"] == [False, False]
    for _ in range(5):
        _, _, _, _, info = env.step([0, 0], [1, 1])
    assert info["msgs_sent"] == [0, 0]

    env = RocketEnv(comm_budget=3)
    obs, _ = env.reset(seed=0)
    assert obs[0][OBS_SLICES["budget"]][0] == 1.0
    for _ in range(6):
        obs, _, _, _, info = env.step([0, 0], [1, 0])
    assert info["msgs_sent"] == [3, 0] and info["can_send"] == [False, True]
    assert obs[0][OBS_SLICES["budget"]][0] == 0.0
    print("ok  communication budget enforced and observable")


def test_comm_period():
    env = RocketEnv(comm_every=5)
    _, info = env.reset(seed=0)
    sent_at = []
    for t in range(1, 16):
        allowed = info["can_send"][0]
        _, _, _, _, info = env.step([0, 0], [1, 1])
        if allowed:
            sent_at.append(t)
    assert sent_at == [5, 10, 15] and info["msgs_sent"] == [3, 3]
    print("ok  periodic communication")


def test_comm_delay():
    for delay in (0, 1, 2):
        env = RocketEnv(comm_delay=delay)
        env.reset(seed=0)
        obs, *_ = env.step([0, 0], [3, 0])
        seen_at = 0 if obs[1][OBS_SLICES["msg_token"]][3] == 1 else None
        for k in range(1, 4):
            if seen_at is not None:
                break
            obs, *_ = env.step([0, 0], [0, 0])
            if obs[1][OBS_SLICES["msg_token"]][3] == 1:
                seen_at = k
        assert seen_at == delay, (delay, seen_at)
        assert obs[0][OBS_SLICES["msg_token"]][0] == 1  # sender receives nothing
    print("ok  message delay")


def test_message_carries_ball_sighting():
    env = RocketEnv(obs_frac=0.25, opponent=False)
    env.reset(seed=0)
    env._agent_pos[:] = [[10.0, 14.0], [10.0, 1.0]]   # agent 0 near ball, agent 1 far
    _freeze_ball(env, [10.0, 16.0])
    obs, *_ = env.step([0, 0], [1, 0])
    assert obs[0][OBS_SLICES["ball_vis"]][0] == 1.0
    assert obs[1][OBS_SLICES["ball_vis"]][0] == 0.0
    assert np.all(obs[1][OBS_SLICES["ball"]] == 0.0)
    assert obs[1][OBS_SLICES["msg_ball_vis"]][0] == 1.0
    assert np.allclose(obs[1][OBS_SLICES["msg_ball"]][:2], obs[0][OBS_SLICES["ball"]])
    assert obs[1][OBS_SLICES["msg_age"]][0] == 0.0
    assert np.allclose(obs[1][OBS_SLICES["msg_sender"]], obs[0][OBS_SLICES["self"]])
    obs, *_ = env.step([0, 0], [0, 0])
    assert obs[1][OBS_SLICES["msg_age"]][0] > 0.0
    print("ok  messages carry the sender's ball sighting under partial observability")


def test_message_expires():
    from env.rocket_env import MSG_TTL
    env = RocketEnv(comm_budget=1)
    env.reset(seed=0)
    obs, *_ = env.step([0, 0], [2, 0])
    assert obs[1][OBS_SLICES["msg_token"]][2] == 1.0
    for _ in range(MSG_TTL):
        obs, *_ = env.step([0, 0], [2, 0])   # budget spent: nothing new is sent
    assert obs[1][OBS_SLICES["msg_token"]][2] == 1.0
    obs, *_ = env.step([0, 0], [0, 0])
    assert obs[1][OBS_SLICES["msg_token"]][0] == 1.0
    assert obs[1][OBS_SLICES["msg_ball_vis"]][0] == 0.0
    assert obs[1][OBS_SLICES["msg_age"]][0] == 1.0
    print("ok  messages expire after MSG_TTL steps")


def test_nstep_accumulator():
    acc = NStepAccumulator(3, 0.5)
    assert acc.push("o0", 0, 1.0, "o1", False, False) == []
    assert acc.push("o1", 0, 2.0, "o2", False, False) == []
    out = acc.push("o2", 0, 4.0, "o3", False, False)
    assert out == [("o0", 0, 1.0 + 0.5 * 2.0 + 0.25 * 4.0, "o3", False, 0.125)]
    out = acc.push("o3", 0, 8.0, "o4", True, False)
    assert [t[0] for t in out] == ["o1", "o2", "o3"] and all(t[4] for t in out)
    assert np.isclose(out[-1][2], 8.0)
    print("ok  n-step returns")


def test_agent_learns_and_messages():
    agent = DQNAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS, batch_size=8, learning_starts=8)
    rng = np.random.default_rng(0)
    before = agent.q_net.params[0].copy()
    for t in range(20):
        o, o2 = rng.normal(size=(2, OBS_DIM)).astype(np.float32)
        agent.store(o, rng.integers(N_ACTIONS), 1.0, o2, terminated=(t == 19))
        if t % 2 == 0:
            agent.message_decision(o, 1)
        agent.message_reward(1.0)
    agent.end_episode(o2)
    assert agent.update() is not None and agent.update_message_head() is not None
    assert not np.array_equal(before, agent.q_net.params[0])
    # first message option spans 2 steps of reward 1
    assert np.isclose(agent.msg_buffer.rew[0], 1 + 0.99)
    assert agent.msg_buffer.done[len(agent.msg_buffer) - 1] == 1.0
    print("ok  DQNAgent updates action and message heads")


def test_evaluate_is_side_effect_free():
    env = RocketEnv(max_steps=30)
    agents = [DQNAgent(OBS_DIM, N_ACTIONS, agent_id=i) for i in range(2)]
    for a in agents:
        a.eps = 0.7
    m1 = evaluate(agents, env, n_eval=3)
    m2 = evaluate(agents, env, n_eval=3)
    assert m1 == m2, "evaluation should be deterministic"
    assert all(a.eps == 0.7 and len(a.buffer) == 0 for a in agents)
    silent = evaluate(agents, env, n_eval=3, silence=True)
    assert silent["msgs_mean"] == 0
    print("ok  evaluation is deterministic and side-effect free")


def test_train_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        h = train(setting="D", n_episodes=4, max_steps=25, save_dir=tmp, eval_every=2,
                  n_eval_episodes=2, n_final_eval=2, verbose=False)
        assert {"final_eval", "final_eval_silenced"} <= set(h)
        with open(os.path.join(tmp, "history.json")) as f:
            assert json.load(f)["eval_episode"] == [2, 4]
        assert os.path.exists(os.path.join(tmp, "agent1_q.npz"))
    print("ok  train() end-to-end")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nAll tests passed!")
