import os
import time
from src.gym_environment import LaneChangeEnv
from src.ql import QLearningAgent
from src.dql import load_drl_agent

SUMO_CFG = os.path.join("sumo_env", "highway.sumocfg")
MODELS_DIR = "models"


def visualize_agent(agent_type, episodes=1):
    print(f"\n--- Visualizing {agent_type} Agent ---")

    mode = "continuous"
    if agent_type == "Q-Learning":
        mode = "discrete"

    env = LaneChangeEnv(SUMO_CFG, mode=mode, use_gui=True)

    agent = None
    if agent_type == "Q-Learning":
        agent = QLearningAgent(action_size=env.action_space.n)
        model_path = os.path.join(MODELS_DIR, "q_table.pkl")
        if os.path.exists(model_path):
            agent.load(model_path)
            agent.epsilon = 0
        else:
            print("Model not found!")
            return

    elif agent_type == "DQN":
        model_path = os.path.join(MODELS_DIR, "dqn_agent")
        if os.path.exists(model_path + ".zip"):
            agent = load_drl_agent(model_path)
        else:
            print("Model not found!")
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
            time.sleep(0.5)

        print(f"Episode finished. Total Reward: {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    # You can comment/uncomment lines below to choose what to watch
    visualize_agent("Q-Learning", episodes=1)
    # visualize_agent('SUMO Default', episodes=1)
    # visualize_agent('DQN', episodes=1)
    # visualize_agent('Random', episodes=1)
