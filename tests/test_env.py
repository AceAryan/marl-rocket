"""Quick smoke tests."""
import sys, os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from env.rocket_env import RocketEnv, N_ACTIONS, N_MESSAGES, SHAPING_SCALE
from agents.dqn_agent import DQNAgent
from experiments.train import evaluate


def test_env_reset():
    env = RocketEnv()
    obs_list, info = env.reset(seed=0)
    assert len(obs_list) == 2
    assert obs_list[0].shape == env.observation_space.shape
    print("✓ env.reset()")


def test_curriculum_modes():
    easy = RocketEnv(curriculum="easy")
    full = RocketEnv(curriculum="full")
    easy.reset(seed=0)
    full.reset(seed=0)
    assert easy.goal_width == 10.0
    assert full.goal_width == 6.0
    assert np.isclose(easy._ball_pos[1], 0.75 * 30.0)

    opponent_before = easy._opp_pos.copy()
    easy._move_opponent()
    assert np.array_equal(easy._opp_pos, opponent_before)

    easy._ball_pos = np.array([14.0, 29.7], dtype=np.float32)
    full._ball_pos = np.array([14.0, 29.7], dtype=np.float32)
    easy_reward, easy_terminated = easy._compute_reward()
    full_reward, full_terminated = full._compute_reward()
    assert easy_reward >= 100.0 and easy_terminated
    assert full_reward < 100.0 and not full_terminated
    print("✓ easy/full curriculum mechanics")


def test_env_step():
    env = RocketEnv()
    obs_list, _ = env.reset(seed=0)
    actions  = [0, 0]
    messages = [1, 2]
    next_obs, reward, term, trunc, info = env.step(actions, messages)
    assert len(next_obs) == 2
    assert isinstance(reward, (float, np.floating))
    assert all(env.observation_space.contains(obs) for obs in next_obs)
    print("✓ env.step()")


def test_potential_reward_shaping():
    env = RocketEnv(max_steps=3, comm_budget=0)
    env.reset(seed=0)
    env._ball_pos = np.array([10.0, 15.0], dtype=np.float32)
    env._ball_vel = np.zeros(2, dtype=np.float32)
    phi_before = env._potential(env._ball_pos)

    _, reward, _, _, info = env.step([0, 0], [0, 0])
    phi_after = env._potential(env._ball_pos)
    expected = SHAPING_SCALE * (0.99 * phi_after - phi_before)
    assert np.isclose(reward, expected)
    assert info["task_reward"] == 0.0
    assert np.isclose(info["shaping_reward"], expected)
    print("✓ potential-based reward shaping")


def test_comm_budget_zero():
    env = RocketEnv(comm_budget=0)
    obs_list, info = env.reset(seed=0)
    assert obs_list[0][-1] == 0.0
    for _ in range(10):
        obs_list, _, term, trunc, info = env.step([1,1], [1,1])
        if term or trunc: break
    assert all(m == 0 for m in info["msgs_sent"]), "Budget=0 but messages were sent"
    print("✓ comm_budget=0 enforced")


def test_remaining_budget_observation():
    env = RocketEnv(comm_budget=10)
    obs_list, _ = env.reset(seed=0)
    assert obs_list[0][-1] == 1.0

    obs_list, _, _, _, _ = env.step([0, 0], [1, 0])
    assert np.isclose(obs_list[0][-1], 0.9)
    assert obs_list[1][-1] == 1.0

    unlimited_env = RocketEnv(comm_budget=None)
    unlimited_obs, _ = unlimited_env.reset(seed=0)
    assert unlimited_obs[0][-1] == 1.0
    print("✓ remaining communication budget is observable")


def test_partial_obs():
    env = RocketEnv(obs_frac=0.25)
    obs_list, _ = env.reset(seed=0)
    assert obs_list[0].shape == env.observation_space.shape
    assert np.all(obs_list[0][4:8] == 0.0)
    assert np.any(obs_list[0][2:4] != 0.0)
    print("✓ partial observability")


def test_observation_space_bounds():
    env = RocketEnv()
    obs_list, _ = env.reset(seed=0)
    for _ in range(100):
        assert all(env.observation_space.contains(obs) for obs in obs_list)
        obs_list, _, terminated, truncated, _ = env.step([5, 1], [0, 0])
        if terminated or truncated:
            obs_list, _ = env.reset()
    print("✓ observations satisfy their declared space")


def test_agent_select():
    obs_dim = RocketEnv().observation_space.shape[0]
    agent = DQNAgent(obs_dim=obs_dim, n_actions=N_ACTIONS, n_messages=N_MESSAGES)
    obs = np.zeros(obs_dim, dtype=np.float32)
    a = agent.select_action(obs)
    m = agent.select_message(obs)
    assert 0 <= a < N_ACTIONS
    assert 0 <= m < N_MESSAGES
    print("✓ DQNAgent.select_action / select_message")


def test_agent_update():
    obs_dim = RocketEnv().observation_space.shape[0]
    agent = DQNAgent(obs_dim=obs_dim, n_actions=N_ACTIONS, batch_size=8)
    observations = np.random.default_rng(0).normal(size=(10, obs_dim)).astype(np.float32)
    first_layer = agent.q_net.net[0].weight.detach().clone()
    second_layer = agent.q_net.net[2].weight.detach().clone()
    for obs in observations:
        agent.store(obs, 0, 1.0, obs, False)
    loss = agent.update()
    assert loss is not None
    assert not np.array_equal(first_layer.numpy(), agent.q_net.net[0].weight.detach().numpy())
    assert not np.array_equal(second_layer.numpy(), agent.q_net.net[2].weight.detach().numpy())
    print(f"✓ DQNAgent.update() loss={loss:.4f}")


def test_goal_replay_buffer():
    obs_dim = RocketEnv().observation_space.shape[0]
    agent = DQNAgent(
        obs_dim=obs_dim,
        n_actions=N_ACTIONS,
        batch_size=8,
        target_update=100,
    )
    obs = np.ones(obs_dim, dtype=np.float32)
    for index in range(16):
        agent.store(obs, 0, 1.0, obs, False, goal_transition=index < 8)
    assert len(agent.buffer) == 16
    assert len(agent.goal_buffer) == 8
    assert agent.update() is not None
    print("✓ goal transitions are retained for replay")


def test_agent_message_update():
    obs_dim = RocketEnv().observation_space.shape[0]
    agent = DQNAgent(obs_dim=obs_dim, n_actions=N_ACTIONS, batch_size=8)
    obs = np.zeros(obs_dim, dtype=np.float32)
    for _ in range(10):
        agent.store_message(obs, 1, 1.0, obs, False)
    loss = agent.update_message_head()
    assert loss is not None
    print(f"✓ DQNAgent.update_message_head() loss={loss:.4f}")


def test_agent_checkpoint_compatibility():
    obs_dim = RocketEnv().observation_space.shape[0]
    original = DQNAgent(obs_dim=obs_dim, n_actions=N_ACTIONS, agent_id=0)
    migrated = DQNAgent(obs_dim=obs_dim, n_actions=N_ACTIONS, agent_id=0)
    with tempfile.TemporaryDirectory() as path:
        for name, net in (("q", original.q_net), ("msg", original.msg_net)):
            np.savez(
                os.path.join(path, f"agent0_{name}.npz"),
                W1=net.net[0].weight.detach().numpy().T,
                b1=net.net[0].bias.detach().numpy(),
                W2=net.net[2].weight.detach().numpy().T,
                b2=net.net[2].bias.detach().numpy(),
                W3=net.net[4].weight.detach().numpy().T,
                b3=net.net[4].bias.detach().numpy(),
            )
        migrated.load(path)

    for source, target in (
        (original.q_net, migrated.q_net),
        (original.msg_net, migrated.msg_net),
    ):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            assert np.array_equal(source_param.detach().numpy(), target_param.detach().numpy())
    print("✓ legacy numpy checkpoints migrate into PyTorch networks")


def test_evaluation_is_read_only():
    env = RocketEnv(max_steps=2, comm_budget=0)
    obs_dim = env.observation_space.shape[0]
    agents = [
        DQNAgent(obs_dim=obs_dim, n_actions=N_ACTIONS, batch_size=8, eps_start=0.4),
        DQNAgent(obs_dim=obs_dim, n_actions=N_ACTIONS, batch_size=8, eps_start=0.7),
    ]
    import random
    random.seed(1234)
    random_state = random.getstate()
    metrics = evaluate(agents, env, n_eval=3, seed=10)
    assert agents[0].eps == 0.4
    assert agents[1].eps == 0.7
    assert all(len(agent.buffer) == 0 for agent in agents)
    assert all(len(agent.msg_buffer) == 0 for agent in agents)
    assert len(metrics) == 5
    assert random.getstate() == random_state
    print("✓ evaluation restores epsilon and does not learn")


def test_short_episode():
    env = RocketEnv(max_steps=50)
    obs_dim = env.observation_space.shape[0]
    agents = [DQNAgent(obs_dim=obs_dim, n_actions=N_ACTIONS, batch_size=8) for _ in range(2)]
    obs_list, _ = env.reset(seed=1)
    total_r = 0.0
    for step in range(50):
        actions  = [agents[i].select_action(obs_list[i]) for i in range(2)]
        messages = [agents[i].select_message(obs_list[i]) for i in range(2)]
        next_obs, r, term, trunc, _ = env.step(actions, messages)
        total_r += r
        for i in range(2):
            agents[i].store(obs_list[i], actions[i], r, next_obs[i], term or trunc)
        obs_list = next_obs
        if term or trunc: break
    print(f"✓ full episode OK, total_reward={total_r:.1f}")


if __name__ == "__main__":
    test_env_reset()
    test_curriculum_modes()
    test_env_step()
    test_potential_reward_shaping()
    test_comm_budget_zero()
    test_remaining_budget_observation()
    test_partial_obs()
    test_observation_space_bounds()
    test_agent_select()
    test_agent_update()
    test_goal_replay_buffer()
    test_agent_message_update()
    test_agent_checkpoint_compatibility()
    test_evaluation_is_read_only()
    test_short_episode()
    print("\n✅ All tests passed!")
