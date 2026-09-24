"""Quick smoke tests."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from env.rocket_env import RocketEnv, N_ACTIONS, N_MESSAGES
from agents.dqn_agent import DQNAgent


def test_env_reset():
    env = RocketEnv()
    obs_list, info = env.reset(seed=0)
    assert len(obs_list) == 2
    assert obs_list[0].shape == env.observation_space.shape
    print("✓ env.reset()")


def test_env_step():
    env = RocketEnv()
    obs_list, _ = env.reset(seed=0)
    actions  = [0, 0]
    messages = [1, 2]
    next_obs, reward, term, trunc, info = env.step(actions, messages)
    assert len(next_obs) == 2
    assert isinstance(reward, (float, np.floating))
    print("✓ env.step()")


def test_comm_budget_zero():
    env = RocketEnv(comm_budget=0)
    obs_list, info = env.reset(seed=0)
    for _ in range(10):
        obs_list, _, term, trunc, info = env.step([1,1], [1,1])
        if term or trunc: break
    assert all(m == 0 for m in info["msgs_sent"]), "Budget=0 but messages were sent"
    print("✓ comm_budget=0 enforced")


def test_partial_obs():
    env = RocketEnv(obs_frac=0.25)
    obs_list, _ = env.reset(seed=0)
    assert obs_list[0].shape == env.observation_space.shape
    print("✓ partial observability")


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
    obs = np.zeros(obs_dim, dtype=np.float32)
    # Fill buffer
    for _ in range(10):
        agent.store(obs, 0, 1.0, obs, False)
    loss = agent.update()
    assert loss is not None
    print(f"✓ DQNAgent.update() loss={loss:.4f}")


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
    test_env_step()
    test_comm_budget_zero()
    test_partial_obs()
    test_agent_select()
    test_agent_update()
    test_short_episode()
    print("\n✅ All tests passed!")
