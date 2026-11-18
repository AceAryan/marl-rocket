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
    assert sent_at == [3, 8, 13] and info["msgs_sent"] == [3, 3]   # phase ceil(k/2), then every k
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
    # The time limit is terminal too (remaining time is observed): no bootstrapping past it
    acc.reset()
    acc.push("a", 0, 1.0, "b", False, False)
    out = acc.push("b", 0, 1.0, "c", False, True)
    assert len(out) == 2 and all(t[4] for t in out)
    print("ok  n-step returns (terminal at the time limit)")


def test_agent_learns_and_gates():
    agent = DQNAgent(obs_dim=OBS_DIM, n_actions=N_ACTIONS, batch_size=8, learning_starts=8)
    agent.gate.learning_starts = 8
    agent.gate.batch_size = 8
    rng = np.random.default_rng(0)
    before_q = agent.q_net.flat.copy()
    before_g = agent.msg_net.flat.copy()
    for t in range(20):
        o, o2 = rng.normal(size=(2, OBS_DIM)).astype(np.float32)
        agent.store(o, rng.integers(N_ACTIONS), 1.0, o2, terminated=(t == 19))
        agent.gate.push(o, 1, 0.5)
    agent.end_episode()
    assert agent.update() is not None and agent.update_gate() is not None
    assert not np.array_equal(before_q, agent.q_net.flat)
    assert not np.array_equal(before_g, agent.msg_net.flat)
    assert agent.learner.buffer.done[len(agent.learner.buffer) - 1] == 1.0
    print("ok  DQNAgent updates its Q-network and message gate")


def test_evaluate_is_side_effect_free():
    from agents.dqn_agent import make_agents
    from experiments.train import make_env
    for setting in ("L4", "Finf"):
        env = make_env(setting, obs_frac=0.25, max_steps=30)
        agents = make_agents(OBS_DIM, N_ACTIONS)
        for a in agents:
            a.eps = 0.7
        kind = "learned" if setting == "L4" else "fixed"
        m1 = evaluate(agents, env, n_eval=3, sender=kind)
        m2 = evaluate(agents, env, n_eval=3, sender=kind)
        assert m1 == m2, "evaluation should be deterministic"
        assert all(a.eps == 0.7 for a in agents) and len(agents[0].learner.buffer) == 0
        assert evaluate(agents, env, n_eval=3, silence=True, sender=kind)["msgs_mean"] == 0
        shuffled = evaluate(agents, env, n_eval=3, shuffle=True, sender=kind)
        assert set(shuffled) == set(m1) and env.payload_hook is None
    print("ok  evaluation is deterministic and side-effect free")


def test_train_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        h = train(setting="D", n_episodes=4, max_steps=25, save_dir=tmp, eval_every=2,
                  n_eval_episodes=2, n_final_eval=2, verbose=False)
        assert {"final_eval", "final_eval_last", "final_eval_shuffled"} <= set(h)
        with open(os.path.join(tmp, "history.json")) as f:
            saved = json.load(f)
        assert saved["eval_episode"] == [2, 4] and saved["label"] == "D" and saved["config"]["seed"] == 42
        assert os.path.exists(os.path.join(tmp, "agent1_q.npz"))
        assert not os.path.exists(os.path.join(tmp, "history.json.tmp"))
    print("ok  train() end-to-end")


def test_memory_keeps_own_sighting():
    from env.rocket_env import MEM_AGE_CAP
    env = RocketEnv(obs_frac=0.15, opponent=False)
    env.reset(seed=0)
    env._agent_pos[:] = [[10.0, 12.0], [2.0, 2.0]]
    _freeze_ball(env, [10.0, 14.0])
    obs, *_ = env.step([0, 0], [0, 0])
    assert obs[0][OBS_SLICES["ball_vis"]][0] == 1.0
    assert np.allclose(obs[0][OBS_SLICES["mem_ball"]], obs[0][OBS_SLICES["ball"]])
    assert obs[0][OBS_SLICES["mem_ball_age"]][0] == 0.0
    env._agent_pos[0] = [10.0, 1.0]                     # walk out of view
    obs, *_ = env.step([0, 0], [0, 0])
    obs, *_ = env.step([0, 0], [0, 0])
    assert obs[0][OBS_SLICES["ball_vis"]][0] == 0.0
    assert np.allclose(obs[0][OBS_SLICES["mem_ball"]], [0.0, 2 * 14.0 / 30.0 - 1.0], atol=1e-5)
    assert np.isclose(obs[0][OBS_SLICES["mem_ball_age"]][0], 2 / MEM_AGE_CAP)
    assert np.allclose(obs[0][OBS_SLICES["mem_ball_rel"]], [0.0, 13.0 / 30.0], atol=1e-4)
    assert obs[1][OBS_SLICES["mem_ball_age"]][0] == 1.0  # agent 1 never saw it
    print("ok  memory keeps the last own sighting with its age")


def test_memory_from_message_uses_send_time():
    from env.rocket_env import MEM_AGE_CAP
    env = RocketEnv(obs_frac=0.15, opponent=False, comm_delay=2)
    env.reset(seed=0)
    env._agent_pos[:] = [[10.0, 14.0], [2.0, 1.0]]
    _freeze_ball(env, [10.0, 16.0])
    obs, *_ = env.step([0, 0], [1, 0])                  # sent at step 1, describing step 0
    assert obs[1][OBS_SLICES["mem_ball_age"]][0] == 1.0
    obs, *_ = env.step([0, 0], [0, 0])
    obs, *_ = env.step([0, 0], [0, 0])                  # delivered at step 3
    assert np.allclose(obs[1][OBS_SLICES["mem_ball"]], obs[0][OBS_SLICES["ball"]])
    assert np.isclose(obs[1][OBS_SLICES["mem_ball_age"]][0], 3 / MEM_AGE_CAP)
    assert np.allclose(obs[1][OBS_SLICES["mem_mate"]], [0.0, 2 * 14.0 / 30.0 - 1.0], atol=1e-5)
    for _ in range(15):                                 # outlives the message (TTL 10)
        obs, *_ = env.step([0, 0], [0, 0])
    assert obs[1][OBS_SLICES["msg_token"]][0] == 1.0
    assert np.isclose(obs[1][OBS_SLICES["mem_ball_age"]][0], 18 / MEM_AGE_CAP)
    print("ok  memory is updated from messages, aged from the send time")


def test_memory_prefers_newer_and_can_be_disabled():
    env = RocketEnv(obs_frac=0.15, opponent=False, comm_delay=3)
    env.reset(seed=0)
    env._agent_pos[:] = [[10.0, 14.0], [2.0, 1.0]]
    _freeze_ball(env, [10.0, 16.0])
    env.step([0, 0], [1, 0])                            # old sighting in flight
    env._agent_pos[1] = [10.0, 13.0]                    # agent 1 now sees the ball itself
    for _ in range(3):
        obs, *_ = env.step([0, 0], [0, 0])
    assert obs[1][OBS_SLICES["mem_ball_age"]][0] == 0.0  # own newer sighting kept
    off = RocketEnv(obs_frac=0.15, memory=False)
    obs, _ = off.reset(seed=0)
    assert np.all(obs[0][OBS_SLICES["mem_ball"]] == 0) and obs[0][OBS_SLICES["mem_ball_age"]][0] == 1.0
    print("ok  memory keeps the newest sighting; memory=False blanks it")


def test_unseen_teammate_touch_is_not_observed_without_a_message():
    from env.rocket_env import ASSIST_WINDOW
    for budget in (0, None):
        env = RocketEnv(obs_frac=0.075, opponent=False, comm_budget=budget)
        env.reset(seed=0)
        env._agent_pos[:] = [[10.0, 14.0], [2.0, 1.0]]   # agent 1 sees neither the ball nor agent 0
        _freeze_ball(env, [10.0, 15.2])
        obs, *_ = env.step([1, 0], [0, 0])               # agent 0 touches, says nothing
        assert env._last_touch[0] == 1
        assert obs[1][OBS_SLICES["assist_me"]][0] == 0.0   # the true touch history is not leaked
        assert obs[0][OBS_SLICES["assist_mate"]][0] == 1.0  # own touch is known
        if budget == 0:
            continue
        obs, *_ = env.step([0, 0], [1, 0])               # agent 0 reports it
        assert obs[1][OBS_SLICES["assist_me"]][0] == 1.0
        assert obs[1][OBS_SLICES["msg_touch"]][0] == 0.0   # reported as of the decision step (touch age 0)
    print("ok  teammate touches are known only when seen or reported")


def test_seen_teammate_touch_is_observed():
    env = RocketEnv(obs_frac=0.25, opponent=False, comm_budget=0)
    env.reset(seed=0)
    env._agent_pos[:] = [[10.0, 14.0], [10.0, 10.0]]      # agent 1 within view of the ball
    _freeze_ball(env, [10.0, 15.2])
    obs, *_ = env.step([1, 0])
    assert obs[1][OBS_SLICES["assist_me"]][0] == 1.0
    print("ok  a teammate touch in view is observed")


def test_assist_flag_matches_the_goal_rule_on_the_next_step():
    from env.rocket_env import ASSIST_WINDOW
    env = RocketEnv(opponent=False)
    env.reset(seed=0)
    env._step_count = 50
    env._last_touch[1] = 50 - ASSIST_WINDOW            # a goal at step 51 would no longer count
    env._known_mate_touch[0] = 50 - ASSIST_WINDOW
    obs = env._get_obs()
    assert obs[0][OBS_SLICES["assist_me"]][0] == 0.0
    env._known_mate_touch[0] = 51 - ASSIST_WINDOW
    assert env._get_obs()[0][OBS_SLICES["assist_me"]][0] == 1.0
    print("ok  assist flags describe the coming step")


def test_assist_shaping_telescopes_when_an_assist_expires():
    from env.rocket_env import ASSIST_GAMMA, ASSIST_WINDOW
    env = RocketEnv(opponent=False)
    env.reset(seed=0)
    env._agent_pos[:] = [[10.0, 10.0], [10.0, 13.0]]
    _freeze_ball(env, [10.0, 11.0])
    total, discount = 0.0, 1.0
    for t in range(ASSIST_WINDOW + 10):
        acts = [1, 2] if t < 2 else [0, 0]
        if t == 1:   # agent 1 meets the ball on its way back
            env._agent_pos[:] = [[2.0, 2.0], [env._ball_pos[0], env._ball_pos[1] + 0.6]]
            env._ball_vel[:] = 0.0
        if t >= 2:   # park everyone away from the ball after the pass
            env._agent_pos[:] = [[2.0, 2.0], [18.0, 2.0]]
            env._ball_vel[:] = 0.0
        phi = env._potential()
        _, _, term, trunc, info = env.step(acts)
        assist_part = info["shaping_reward"] - (SHAPING_GAMMA * env._potential() - phi)
        total += discount * assist_part
        discount *= ASSIST_GAMMA
    assert env._last_touch[0] == 1 and env._last_touch[1] == 2
    assert not env._assisted(1)                        # the assist has expired
    assert abs(total) < 1e-3, total                    # created and expired: nets zero
    print("ok  assist shaping is charged back when the assist expires")


def test_message_carries_the_decision_time_view():
    env = RocketEnv(obs_frac=0.075, opponent=False)
    obs, _ = env.reset(seed=0)
    env._agent_pos[:] = [[10.0, 13.0], [2.0, 1.0]]
    _freeze_ball(env, [10.0, 15.0])                    # 2.0 away: in view (radius 2.7)
    obs = env._get_obs()
    assert obs[0][OBS_SLICES["ball_vis"]][0] == 1.0
    obs, *_ = env.step([2, 0], [1, 0])                 # walks away (out of view) while sending
    assert obs[0][OBS_SLICES["ball_vis"]][0] == 0.0
    assert obs[1][OBS_SLICES["msg_ball_vis"]][0] == 1.0
    assert np.allclose(obs[1][OBS_SLICES["msg_ball"]][:2], [0.0, 0.0], atol=1e-6)   # (10, 15) normalised
    assert np.allclose(obs[1][OBS_SLICES["msg_sender"]], [0.0, 2 * 13.0 / 30.0 - 1.0], atol=1e-6)
    print("ok  a message reports what the sender saw when it decided to send")


def test_fixed_schedule_and_last_useful_send():
    for every, delay, expected in ((40, 0, [20, 60]), (16, 0, [8, 24, 40, 56, 72]), (1, 0, list(range(1, 80))),
                                   (1, 3, list(range(1, 77)))):
        env = RocketEnv(comm_every=every, comm_delay=delay, opponent=False, assist_required=False)
        _, info = env.reset(seed=0)
        env._ball_pos[:] = [1.0, 15.0]
        sent_at = []
        for t in range(1, 81):
            if info["can_send"][0]:
                sent_at.append(t)
            _, _, term, trunc, info = env.step([0, 0], [1, 0])
            if term or trunc:
                break
        assert sent_at == expected, (every, delay, sent_at)
    print("ok  fixed schedules are phased and never send what cannot be used")


def test_budget_features():
    from env.rocket_env import BUDGET_CAP
    import math
    env = RocketEnv(comm_budget=4, opponent=False)
    obs, _ = env.reset(seed=0)
    for i in range(2):
        assert obs[i][OBS_SLICES["budget"]][0] == 1.0
        assert np.isclose(obs[i][OBS_SLICES["budget_left"]][0], math.log1p(4) / math.log1p(BUDGET_CAP))
    obs, *_ = env.step([0, 0], [1, 0])
    expect = math.log1p(3) / math.log1p(BUDGET_CAP)
    assert np.isclose(obs[0][OBS_SLICES["budget"]][0], 0.75)
    assert np.isclose(obs[0][OBS_SLICES["budget_left"]][0], expect)
    assert np.isclose(obs[1][OBS_SLICES["mate_budget_left"]][0], expect)   # inferred from the delivery
    assert np.isclose(obs[0][OBS_SLICES["mate_budget_left"]][0], math.log1p(4) / math.log1p(BUDGET_CAP))
    obs, _ = RocketEnv(comm_budget=None).reset(seed=0)
    assert all(obs[0][OBS_SLICES[k]][0] == 1.0 for k in ("budget", "budget_left", "mate_budget_left"))
    obs, _ = RocketEnv(comm_budget=0).reset(seed=0)
    assert all(obs[0][OBS_SLICES[k]][0] == 0.0 for k in ("budget", "budget_left", "mate_budget_left"))
    blind = RocketEnv(comm_budget=4, budget_obs=False)
    obs, _ = blind.reset(seed=0)
    obs, *_ = blind.step([0, 0], [1, 1])
    assert all(o[OBS_SLICES[k]][0] == 0.0 for o in obs for k in ("budget", "budget_left", "mate_budget_left"))
    print("ok  own and teammate budget are observable (and can be hidden)")


def test_clone_replays_identically():
    rng = np.random.default_rng(3)
    for match in (False, True):
        env = RocketEnv(obs_frac=0.15, comm_budget=5, opponent=False, match=match, max_steps=200)
        env.reset(seed=1)
        for _ in range(30):
            env.step(rng.integers(N_ACTIONS, size=2).tolist(), [1, 0])
        twin = env.clone()
        if match:   # force a goal so the kickoff draws from the RNG
            for e in (env, twin):
                e._last_touch[1] = e._step_count
                e._agent_pos[:] = [[2.0, 2.0], [18.0, 2.0]]
                _freeze_ball(e, [FIELD_W / 2, FIELD_H - 1.0], [0.0, 1.0])
                e._last_toucher = 0
        actions = rng.integers(N_ACTIONS, size=(60, 2)).tolist()
        for a in actions:
            o1, r1, t1, u1, i1 = env.step(a, [1, 1])
            o2, r2, t2, u2, i2 = twin.step(a, [1, 1])
            assert all(np.array_equal(x, y) for x, y in zip(o1, o2)) and r1 == r2 and (t1, u1) == (t2, u2)
            if t1 or u1:
                break
        if match:
            assert env._score_team == twin._score_team == 1
    print("ok  clone() reproduces the trajectory, RNG included")


def test_match_mode_restarts_play_after_a_goal():
    env = RocketEnv(obs_frac=0.15, comm_budget=5, opponent=False, match=True, max_steps=200)
    env.reset(seed=0)
    for _ in range(3):
        env.step([0, 0], [1, 1])
    sent = list(env._msgs_sent)
    env._agent_pos[:] = [[2.0, 2.0], [18.0, 2.0]]
    env._last_touch[1] = env._step_count
    env._last_toucher = 0
    _freeze_ball(env, [FIELD_W / 2, FIELD_H - 1.0], [0.0, 1.0])
    obs, reward, term, trunc, info = env.step([0, 0], [1, 1])
    assert not term and not trunc and info["goals"] == 1 and info["goal_steps"] == [4]
    assert info["task_reward"] == GOAL_REWARD
    assert env._msgs_sent == [s + 1 for s in sent]          # budget carries on (the goal-step send counts)
    assert all(p[1] <= 0.35 * FIELD_H for p in env._agent_pos)
    assert 0.4 * FIELD_H <= env._ball_pos[1] <= 0.6 * FIELD_H
    assert env._last_touch == [-10**9, -10**9] and env._known_mate_touch == [-10**9, -10**9]
    for i in range(2):   # messages describing the finished play are discarded
        assert obs[i][OBS_SLICES["msg_token"]][0] == 1.0
        if obs[i][OBS_SLICES["ball_vis"]][0] == 0.0:
            assert obs[i][OBS_SLICES["mem_ball_age"]][0] == 1.0
    while not trunc:
        obs, reward, term, trunc, info = env.step([0, 0])
        assert not term
    assert info["step"] == 200
    print("ok  match mode: a goal restarts play, the clock and budget carry on")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nAll tests passed!")
