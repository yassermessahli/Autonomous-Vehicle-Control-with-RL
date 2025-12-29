from stable_baselines3 import DQN
import os


def train_drl_agent(env, total_timesteps=10000, log_dir="logs", model_path="models/dqn_agent"):
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    print("Starting DQN Training...")
    model.learn(total_timesteps=total_timesteps)
    print("Training Finished.")

    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model


def load_drl_agent(model_path):
    if os.path.exists(model_path + ".zip"):
        return DQN.load(model_path)
    else:
        print(f"Model not found at {model_path}")
        return None
