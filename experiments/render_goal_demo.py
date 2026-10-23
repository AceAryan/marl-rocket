"""Render a ball bounce and a goal with hardcoded upward actions."""

import numpy as np

from env.rocket_env import BALL_RADIUS, FIELD_H, FIELD_W, RocketEnv


def run_scenario(label: str, ball_x: float) -> tuple[int, bool]:
    env = RocketEnv(curriculum="easy", max_steps=10, render_mode="human")
    env.reset(seed=0)
    env._ball_pos = np.array([ball_x, 25.0], dtype=np.float32)
    env._ball_vel = np.zeros(2, dtype=np.float32)
    env._agent_pos[0] = np.array([ball_x, 24.0], dtype=np.float32)
    env._agent_pos[1] = np.array([2.0, 2.0], dtype=np.float32)
    env._opp_pos = np.array([19.0, 25.5], dtype=np.float32)

    print(f"\n=== {label} (actions: up, noop) ===")
    bounced_off_top = False
    for _ in range(env.max_steps):
        _, _, terminated, truncated, info = env.step([1, 0])
        env.render()
        print(
            f"ball={env._ball_pos.tolist()}, "
            f"velocity={env._ball_vel.tolist()}, "
            f"score={info['score_team']}-{info['score_opp']}"
        )
        bounced_off_top |= (
            env._ball_pos[1] == FIELD_H - BALL_RADIUS and env._ball_vel[1] < 0
        )
        if terminated or truncated:
            break

    return info["score_team"], bounced_off_top


def main() -> None:
    bounce_score, bounced_off_top = run_scenario(
        "Ceiling bounce outside the goal", ball_x=2.0
    )
    goal_score, _ = run_scenario("Aligned top goal", ball_x=FIELD_W / 2)

    assert bounce_score == 0
    assert bounced_off_top
    assert goal_score == 1


if __name__ == "__main__":
    main()
