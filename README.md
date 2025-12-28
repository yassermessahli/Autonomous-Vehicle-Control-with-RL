# Reinforcement Learning for Lane-Changing Decisions on Highways

This project implements an Autonomous Vehicle (AV) agent that learns to make optimal lane-changing decisions on a two-lane highway using Reinforcement Learning. The environment is simulated using SUMO (Simulation of Urban Mobility).

## Project Goal
Train an AV to navigate around motionless obstacle vehicles while maximizing travel time and avoiding collisions.

## Agents Implemented
1.  **Q-Learning Agent**: A tabular RL agent using a discretized state space.
2.  **Deep Q-Network (DQN) Agent**: A Deep RL agent using Stable Baselines 3 and continuous state space.
3.  **SUMO Default Controller**: Baseline comparison using SUMO's built-in lane-changing model.

## Requirements
*   **Python 3.8+**
*   **SUMO**: Must be installed and added to your system PATH.
    *   Download: [https://eclipse.dev/sumo/](https://eclipse.dev/sumo/)
    *   Ensure SUMO_HOME environment variable is set.

## Installation

1.  Clone the repository.
2.  Install Python dependencies:
    `ash
    pip install -r requirements.txt
    `

## Setup Environment

1.  Generate the SUMO configuration files:
    `ash
    python create_env.py
    `
    *   **Note**: If 
etconvert is not in your PATH, the script will fail to generate highway.net.xml. You must generate it manually using the created .nod.xml and .edg.xml files:
        `ash
        netconvert --node-files sumo_env/highway.nod.xml --edge-files sumo_env/highway.edg.xml -o sumo_env/highway.net.xml
        `

## Running the Training & Evaluation

Run the main script to train both agents and evaluate them:
`ash
python main.py
`

This script will:
1.  Train the Q-Learning agent (saved to models/q_table.pkl).
2.  Train the DQN agent (saved to models/dqn_agent.zip).
3.  Evaluate Q-Learning, DQN, Random, and SUMO Default agents.
4.  Save results to logs/evaluation_results.csv.
5.  Generate plots in plots/.

## Project Structure
*   src/: Source code for agents and environment.
    *   nv.py: Custom Gym wrapper for SUMO.
    *   q_learning.py: Q-Learning implementation.
    *   drl_agent.py: DQN wrapper.
*   sumo_env/: Generated SUMO configuration files.
*   models/: Saved trained models.
*   logs/: TensorBoard logs and CSV results.
*   plots/: Evaluation plots.
*   create_env.py: Script to generate SUMO network.
*   main.py: Main execution script.
