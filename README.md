# Autonomous Vehicle Control with RL

This project trains an autonomous vehicle to change lanes and avoid obstacles using Reinforcement Learning (Q-Learning and DQN) in the SUMO traffic simulator.

![Simulation Preview](https://raw.githubusercontent.com/yassermessahli/Autonomous-Vehicle-Control-with-RL/refs/heads/main/assets/sumo.png)

## Project Structure

- **src/**: Contains all source code and scripts.
- **models/**: Stores trained model files.
- **sumo_env/**: Contains SUMO configuration and network files.
- **runs/**: Stores TensorBoard logs.

## How to Run

### 1. Install Dependencies

Make sure you have [**Python 3.13**](https://www.python.org/ftp/python/3.13.0/python-3.13.0-amd64.exe) and [**SUMO**](https://sumo.dlr.de/releases/1.25.0/sumo-win64-1.25.0.msi) installed.

```bash
# sync python dependencies using uv
uv sync
```

### 2. Build Road Network

First, generate the highway environment files:

```bash
python src/create_sumo_network.py
```

### 3. Train Agents

**Train the Q-Learning agent:**

```bash
python src/train_ql.py
```

_Saves to `models/q_table_final.pkl`_

**Train the Deep Q-Network (DQN) agent:**

```bash
python src/train_dql.py
```

_Saves to `models/dqn_final.zip`_

### 4. Run Simulation

Once trained, you can watch the agent drive!

**For DQN:**

```bash
python src/simulate.py --type DQN --model models/dqn_final.zip
```

**For Q-Learning:**

```bash
python src/simulate.py --type Q-Learning --model models/q_table_final.pkl
```

## View Results

To see graphs of the metrics history during training with 1TensorBoard`, run :

```bash
tensorboard --logdir ./runs
```

Then open the provided URL (usually http://localhost:6006) in the browser.

You can also see the results during the past training through my WeightsAndBiases dashboard online. through these links:

- [Q-Learning Dashboard](https://wandb.ai/yassermessahli0-estin/sumo-lane-change-ql?nw=nwuseryassermessahli0)
- [DQN Dashboard](https://wandb.ai/yassermessahli0-estin/sumo-lane-change-dql/workspace?nw=nwuseryassermessahli0)
