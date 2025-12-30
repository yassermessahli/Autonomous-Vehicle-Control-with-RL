import os
from dotenv import load_dotenv
import wandb

from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import DQN

# import utils & environment
from environment import LaneChangeEnv


# load W&B API key from .env
load_dotenv()


def train():
    # Hyperparameters
    config = {
        "total_timesteps": 30000,
        "learning_rate": 0.0001,
        "buffer_size": 50000,
        "learning_starts": 1000,
        "batch_size": 64,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 1000,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.05,
        "env_mode": "continuous",  # DQN usually works better with continuous state space
        "policy": "MlpPolicy",
    }

    # Initialize a new W&B run
    run = wandb.init(
        project="sumo-lane-change-dql",
        config=config,
        sync_tensorboard=True,  # Auto-log SB3 metrics
        monitor_gym=True,  # Auto-log Gym video/metrics
        save_code=True,
    )

    # Initialize Road Environment
    sumo_cfg_path = "sumo_env/highway.sumocfg"

    # Create the environment
    env = LaneChangeEnv(sumo_cfg_file=sumo_cfg_path, mode=config["env_mode"])

    # Initialize the Agent (DQN)
    agent = DQN(
        policy=config["policy"],
        env=env,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        train_freq=config["train_freq"],
        gradient_steps=config["gradient_steps"],
        target_update_interval=config["target_update_interval"],
        exploration_fraction=config["exploration_fraction"],
        exploration_final_eps=config["exploration_final_eps"],
        tensorboard_log=f"runs/DQN/{run.id}",
        verbose=1,
    )

    print(" ==> Training started... Check W&B dashboard for live graphs.")

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Train the agent
    # WandbCallback handles logging of reward, episode length, etc. automatically
    agent.learn(total_timesteps=config["total_timesteps"], callback=WandbCallback())

    # Save Final Model
    agent.save(os.path.join("models", "dqn_final.zip"))

    # Close environment and W&B
    env.close()
    wandb.finish()

    print("✅ Training complete.")


if __name__ == "__main__":
    train()
