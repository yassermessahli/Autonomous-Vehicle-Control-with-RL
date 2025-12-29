import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.gym_environment import LaneChangeEnv
from src.ql import QLearningAgent
from src.dql import train_drl_agent, load_drl_agent

SUMO_CFG = os.path.join("sumo_env", "highway.sumocfg")
MODELS_DIR = "models"
LOGS_DIR = "logs"
PLOTS_DIR = "plots"

# Set to True to see the simulation (will be slower)
USE_GUI = False


def train_q_learning(episodes=100):
    print("--- Training Q-Learning Agent ---")
    env = LaneChangeEnv(SUMO_CFG, mode="discrete", use_gui=USE_GUI)
    agent = QLearningAgent(action_size=env.action_space.n)

    rewards_history = []

    pbar = tqdm(range(episodes), desc="Training Episodses")
    for e in pbar:
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)
        # pbar.set_postfix({'Reward': f'{total_reward:.2f}', 'Epsilon': f'{agent.epsilon:.2f}'})

    agent.save(os.path.join(MODELS_DIR, "q_table.pkl"))
    env.close()

    # Plot Training Curve
    plt.figure()
    plt.plot(rewards_history)
    plt.title("Q-Learning Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(os.path.join(PLOTS_DIR, "q_learning_training.png"))
    print("Q-Learning Training Completed.")


def train_drl(timesteps=10000):
    print("--- Training Deep RL Agent (DQN) ---")
    env = LaneChangeEnv(SUMO_CFG, mode="continuous", use_gui=USE_GUI)
    # Stable Baselines 3 has its own progress bar if verbose=1
    train_drl_agent(
        env, total_timesteps=timesteps, log_dir=LOGS_DIR, model_path=os.path.join(MODELS_DIR, "dqn_agent")
    )
    env.close()
    print("DRL Training Completed.")


def evaluate(episodes=10):
    print("--- Evaluating Agents ---")
    results = []

    # 1. Evaluate Q-Learning
    env_q = LaneChangeEnv(SUMO_CFG, mode="discrete", use_gui=USE_GUI)
    agent_q = QLearningAgent(action_size=env_q.action_space.n)
    agent_q.load(os.path.join(MODELS_DIR, "q_table.pkl"))
    agent_q.epsilon = 0  # Greedy

    for e in tqdm(range(episodes), desc="Eval Q-Learning"):
        state, _ = env_q.reset()
        total_reward = 0
        done = False
        steps = 0
        collision = 0

        while not done:
            action = agent_q.get_action(state)
            next_state, reward, terminated, truncated, _ = env_q.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            if terminated and reward < -40:  # Assuming collision penalty is high
                collision = 1
            state = next_state

        results.append(
            {
                "Agent": "Q-Learning",
                "Episode": e,
                "Reward": total_reward,
                "Collision": collision,
                "Steps": steps,
            }
        )
    env_q.close()

    # 2. Evaluate DRL
    env_drl = LaneChangeEnv(SUMO_CFG, mode="continuous", use_gui=USE_GUI)
    agent_drl = load_drl_agent(os.path.join(MODELS_DIR, "dqn_agent"))

    for e in tqdm(range(episodes), desc="Eval DQN"):
        obs, _ = env_drl.reset()
        total_reward = 0
        done = False
        steps = 0
        collision = 0

        while not done:
            action, _ = agent_drl.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env_drl.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            if terminated and reward < -40:
                collision = 1

        results.append(
            {"Agent": "DQN", "Episode": e, "Reward": total_reward, "Collision": collision, "Steps": steps}
        )
    env_drl.close()

    # 3. Evaluate Random (Baseline)
    env_rand = LaneChangeEnv(SUMO_CFG, mode="continuous", use_gui=USE_GUI)
    for e in tqdm(range(episodes), desc="Eval Random"):
        obs, _ = env_rand.reset()
        total_reward = 0
        done = False
        steps = 0
        collision = 0

        while not done:
            action = env_rand.action_space.sample()
            obs, reward, terminated, truncated, _ = env_rand.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            if terminated and reward < -40:
                collision = 1

        results.append(
            {"Agent": "Random", "Episode": e, "Reward": total_reward, "Collision": collision, "Steps": steps}
        )
    env_rand.close()

    # 4. Evaluate SUMO Default
    env_sumo = LaneChangeEnv(SUMO_CFG, mode="continuous", use_gui=USE_GUI)
    env_sumo.set_sumo_control(True)

    for e in tqdm(range(episodes), desc="Eval SUMO Default"):
        obs, _ = env_sumo.reset()
        total_reward = 0
        done = False
        steps = 0
        collision = 0

        while not done:
            # Action doesn't matter if sumo_control is True, but we pass 0
            obs, reward, terminated, truncated, _ = env_sumo.step(0)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            if terminated and reward < -40:
                collision = 1

        results.append(
            {
                "Agent": "SUMO Default",
                "Episode": e,
                "Reward": total_reward,
                "Collision": collision,
                "Steps": steps,
            }
        )
    env_sumo.close()

    # Save Results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(LOGS_DIR, "evaluation_results.csv"), index=False)
    print("Evaluation Completed. Results saved.")

    # Plot Comparison
    plt.figure()
    df.groupby("Agent")["Reward"].mean().plot(kind="bar", yerr=df.groupby("Agent")["Reward"].std())
    plt.title("Average Reward per Agent")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(PLOTS_DIR, "evaluation_reward.png"))

    plt.figure()
    df.groupby("Agent")["Collision"].mean().plot(kind="bar")
    plt.title("Collision Rate per Agent")
    plt.ylabel("Collision Rate")
    plt.savefig(os.path.join(PLOTS_DIR, "evaluation_collision.png"))


if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1. Train Q-Learning
    train_q_learning(episodes=50)  # Short training for demo

    # 2. Train DRL
    train_drl(timesteps=5000)  # Short training for demo

    # 3. Evaluate
    evaluate(episodes=10)
