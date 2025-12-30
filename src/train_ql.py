import os
import wandb
from tqdm import tqdm
from dotenv import load_dotenv
from torch.utils.tensorboard import SummaryWriter  # <--- 1. Import SummaryWriter

# import utils
from utils import suppress_sumo_output

# environment and agent
from environment import LaneChangeEnv
from ql import QLearningAgent

# load W&B API key from .env
load_dotenv()


def train():
    # Hyperparameters
    config = {
        "episodes": 1000,
        "learning_rate": 0.05,
        "gamma": 0.95,
        "epsilon_start": 1.0,
        "epsilon_decay": 0.99,
        "epsilon_min": 0.05,
        "env_mode": "discrete",
    }

    # Initialize Weights & Biases
    # We capture the run object to ensure we get the ID correctly
    run = wandb.init(project="sumo-lane-change-ql", config=config)

    # Setup TensorBoard with W&B Run ID
    tb_ql_log_dir = os.path.join("runs/QL", run.id)
    writer = SummaryWriter(log_dir=tb_ql_log_dir)

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

    print("==> Training started... Check W&B dashboard for live graphs.")

    os.makedirs("models", exist_ok=True)

    # Training Loop
    for episode in tqdm(range(config["episodes"]), desc="Episodes"):

        with suppress_sumo_output():
            state, _ = env.reset()

        total_reward = 0
        steps = 0
        terminated = False
        truncated = False
        collisions = 0
        lane_changes = 0

        while not (terminated or truncated):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if action != 0:
                lane_changes += 1

            if terminated and reward <= -40:
                collisions += 1

        # -------------------------------------------------------
        # 3. Log to TensorBoard
        # -------------------------------------------------------
        # Use add_scalar(tag, value, step)
        writer.add_scalar("Episode/Total Reward", total_reward, episode)
        writer.add_scalar("Episode/Epsilon", agent.epsilon, episode)
        writer.add_scalar("Episode/Steps", steps, episode)
        writer.add_scalar("Episode/Lane Changes", lane_changes, episode)
        writer.add_scalar("Episode/Collisions", collisions, episode)
        # -------------------------------------------------------

        # Log to W&B
        wandb.log(
            data={
                "total reward in episode": total_reward,
                "epsilon value": agent.epsilon,
                "steps in episode": steps,
                "Nb of lane changes": lane_changes,
                "Nb of collisions": collisions,
            },
            step=episode,
        )

    # Save Model
    agent.save(os.path.join("models", "q_table_final.pkl"))

    # 4. Close Writer, Environment and W&B instances
    writer.close()
    env.close()
    wandb.finish()

    print("✅ Training complete.")


if __name__ == "__main__":
    train()
