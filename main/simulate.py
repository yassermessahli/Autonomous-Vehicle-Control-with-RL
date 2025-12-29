import os
import sys
import time
import argparse

# Add project root to path so we can import 'src'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import LaneChangeEnv
from src.ql import QLearningAgent
from stable_baselines3 import DQN

SUMO_CFG = os.path.join("sumo_env", "highway.sumocfg")


def visualize_agent(agent_type, model_path=None, episodes=1):
    print(f"\n--- Visualizing {agent_type} Agent ---")
    if model_path:
        print(f"Loading model from: {model_path}")

    mode = "continuous"
    if agent_type == "Q-Learning":
        mode = "discrete"

    env = LaneChangeEnv(SUMO_CFG, mode=mode, use_gui=True)

    agent = None
    if agent_type == "Q-Learning":
        agent = QLearningAgent(action_size=env.action_space.n)
        if model_path and os.path.exists(model_path):
            agent.load(model_path)
            agent.epsilon = 0
        else:
            print(f"Error: Model not found at {model_path}")
            return

    elif agent_type == "DQN":
        if model_path and os.path.exists(model_path):
            agent = DQN.load(model_path)
        else:
            print(f"Error: Model not found at {model_path}")
            return

    elif agent_type == "SUMO Default":
        env.set_sumo_control(True)

    for e in range(episodes):
        print(f"Episode {e+1}/{episodes}")
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = 0
            if agent_type == "Q-Learning":
                action = agent.get_action(obs)
            elif agent_type == "DQN":
                action, _ = agent.predict(obs, deterministic=True)
            elif agent_type == "Random":
                action = env.action_space.sample()
            # SUMO Default ignores action

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Slow down slightly to make it easier to watch
            time.sleep(0.3)

        print(f"Episode finished. Total Reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trained agents.")
    parser.add_argument(
        "--type",
        type=str,
        default="Q-Learning",
        choices=["Q-Learning", "DQN", "SUMO Default", "Random"],
        help="Type of agent to visualize",
    )
    parser.add_argument(
        "--model", type=str, default="models/q_table_final.pkl", help="Path to the model file"
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")

    args = parser.parse_args()

    visualize_agent(args.type, args.model, args.episodes)
