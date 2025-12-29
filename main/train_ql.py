import gymnasium as gym
import numpy as np
import os
import wandb
from tqdm import tqdm
from dotenv import load_dotenv

# importat utils
from src.utils import suppress_sumo_output

# environment and agent
from src.environment import LaneChangeEnv
from src.ql import QLearningAgent

# load W&B API key from .env
load_dotenv()


# The Training Loop
def train():
    # Hyperparameters
    config = {
        "episodes": 500,
        "learning_rate": 0.1,  # tabular needs higher LR than DQN
        "gamma": 0.95,  # Discount factor
        "epsilon_start": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "env_mode": "discrete",  # Q-Learning requires discrete state space
    }

    # Initialize Weights & Biases
    wandb.init(project="sumo-lane-change", config=config)

    # Initialize Road Environment
    sumo_cfg_path = "sumo_env/highway.sumocfg"
    env = LaneChangeEnv(sumo_cfg_file=sumo_cfg_path, mode=config["env_mode"])

    # Initialize the Agent
    agent = QLearningAgent(
        action_size=env.action_space.n,
        learning_rate=config["learning_rate"],
        discount_factor=config["gamma"],
        epsilon=config["epsilon_start"],
        epsilon_decay=config["epsilon_decay"],
        epsilon_min=config["epsilon_min"],
    )

    print("🚀 Training started... Check W&B dashboard for live graphs.")

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Training Loop
    for episode in tqdm(range(config["episodes"]), desc="Episodes"):

        # reset env (with Silencer)
        with suppress_sumo_output():
            state, _ = env.reset()

        total_reward = 0
        steps = 0
        terminated = False
        truncated = False

        # Metrics for this episode
        collisions = 0
        lane_changes = 0

        while not (terminated or truncated):
            # 1. Get Action
            action = agent.get_action(state)

            # 2. Step Environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            # 3. Learn
            done = terminated or truncated
            agent.learn(state, action, reward, next_state, done)

            # 4. Update State
            state = next_state
            total_reward += reward
            steps += 1

            # Track Metrics
            if action != 0:
                lane_changes += 1

            # Check for collision (based on reward and termination cause)
            # In the env, a crash usually gives -50 reward and terminates
            if terminated and reward <= -40:
                collisions += 1

        # Log to W&B
        wandb.log(
            {
                "total_reward": total_reward,
                "epsilon": agent.epsilon,
                "steps": steps,
                "lane_changes": lane_changes,
                "collisions": collisions,
            }
        )

    # Save Model
    agent.save(os.path.join("models", "q_table_final.pkl"))

    # Close environment and W&B
    env.close()
    wandb.finish()

    print("✅ Training complete.")


if __name__ == "__main__":
    train()
