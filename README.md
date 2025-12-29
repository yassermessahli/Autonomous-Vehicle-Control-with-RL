# Autonomous Vehicle Control with RL

A simple project to train an autonomous vehicle to change lanes, avoid obstacles, and maximize speed using Reinforcement Learning in the SUMO traffic simulator.

![Simulation Preview](placeholder_image.png)

## 📂 Project Structure

- **`main/`**: The scripts to run.
  - `create_sumo_network.py`: Generates the road network files (used by SUMO software and gymnasium).
  - `train_ql.py`: Trains a Q-Learning agent from scratch.
  - `train_dql.py`: Trains a Deep Q-Network (DQN) agent using **Stable Baselines3**.
  - `simulate.py`: Run a simulation to experiment with the trained agents.
- **`src/`**: The module files of the core logic of the agents and the environment.
- **`models/`**: Where the model checkpoints are saved.
- **`sumo_env/`**: Configuration files for the SUMO simulator (auto-generated).

---

## 🚀 How to Run

### 1. Prerequisites

Make sure you have [**Python 3.13**](https://www.python.org/ftp/python/3.13.0/python-3.13.0-amd64.exe) and [**SUMO**](https://sumo.dlr.de/releases/1.25.0/sumo-win64-1.25.0.msi) installed.

```bash
# sync python dependencies
uv sync
```

### 2. Build the Road Network

First, generate the highway environment files:

```bash
python main/create_sumo_network.py
```

### 3. Train the Agents

You can train either a simple Q-Learning agent or a Deep Q-Network agent. We used [Weights & Biases](https://wandb.ai/) to log training metrics in real-time during training to see the progress. So make sure to create a free account and generate an API key and save it in the `.env` file before training.

**To train Q-Learning:**

```bash
python main/train_ql.py
```

_Saves model to `models/q_table_final.pkl`_

**To train DQN:**

```bash
python main/train_dql.py
```

_Saves model to `models/dqn_final.zip`_

### 4. Watch the Simulation

Once trained, you can watch the agent drive!

**Watch DQN:**

```bash
python main/simulate.py --type DQN --model models/dqn_final.zip
```

**Watch Q-Learning:**

```bash
python main/simulate.py --type Q-Learning --model models/q_table_final.pkl
```

---

## 📈 View Results

To see graphs of the metrics history during training:

```bash
tensorboard --logdir runs
```

Then open the link (usually `http://localhost:6006`) in the browser.
