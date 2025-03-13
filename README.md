# AI-Powered Grid World Reinforcement Learning Project

## Overview

This project implements and visualizes various reinforcement learning algorithms in customizable grid-based environments. It serves as both an educational tool for understanding core RL concepts and a platform for experimenting with different algorithms and environment configurations.

The project features a comprehensive implementation of:
- Dynamic Programming methods (Value Iteration, Policy Iteration)
- Temporal Difference Learning (Q-Learning, SARSA)
- Deep Reinforcement Learning (Deep Q-Network)

All within interactive, customizable grid world environments with rich visualization capabilities.

![Grid World Example](https://i.imgur.com/placeholder.jpg)

## Features

- **Flexible Grid Environment**:
  - Variable grid dimensions
  - Different cell types (empty spaces, walls, goals, traps)
  - Deterministic or stochastic transition dynamics
  - Customizable reward structures

- **Multiple RL Algorithms**:
  - Value Iteration and Policy Iteration (model-based)
  - Q-Learning and SARSA (model-free)
  - Deep Q-Network (DQN) with experience replay and target networks

- **Visualization**:
  - Terminal-based grid rendering
  - Interactive Pygame visualization
  - Policy and value function visualization
  - Learning curve and performance metrics
  - Animation of agent trajectories

- **Game Modes**:
  - Create custom grids
  - Use predefined grid templates
  - Generate large-scale environments
  - Interactive play with trained agents

## Installation

### Requirements

- Python 3.6+
- NumPy
- Matplotlib
- Pygame
- TensorFlow (for DQN)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/grid-world-rl.git
cd grid-world-rl
```

2. Install dependencies:
```bash
pip install numpy matplotlib pygame tensorflow
```

## Usage

### Basic Usage

Run the main script to start the interactive game:

```bash
python main.py
```

This provides a text-based interface to:
- Create custom grid worlds or select predefined ones
- Train different RL agents (Value Iteration, Policy Iteration, Q-Learning, SARSA, DQN)
- Play the grid world game with agent guidance

### Enhanced Visualization

For enhanced visualization with Pygame:

```bash
python ai_play.py
```

Command-line options:
```
--custom         Create a custom grid
--predefined     Use a predefined grid
--interactive    Play the game yourself
--load FILE      Load map from pickle file
--no-train       Skip training an agent (for large maps)
```

Example with a custom grid:
```bash
python ai_play.py --custom
```

### Large Map Generation

Generate large grid worlds for more complex environments:

```bash
python large_map_generator.py --size 100 --walls 0.2 --output large_grid.pkl
```

Then load and visualize:
```bash
python ai_play.py --load large_grid.pkl
```

## Controls

### Pygame Visualization Controls

- **SPACE**: Start/Pause simulation
- **R**: Reset the environment
- **+/-**: Increase/decrease simulation speed
- **Q**: Quit the application

### Interactive Game Controls

- **W**: Move Up
- **A**: Move Left
- **S**: Move Down
- **D**: Move Right
- **Q**: Quit the game

### Large Map Controls

- **WASD**: Move camera
- **Mouse Wheel**: Zoom in/out
- **C**: Center camera on agent

## Project Structure

- `GridWorldEnvironment.py`: Core environment class with dynamics, rendering, and rewards
- `dynamic_programming.py`: Value Iteration and Policy Iteration implementations
- `td_learning.py`: Q-Learning and SARSA implementations
- `deep_rl.py`: Deep Q-Network implementation with TensorFlow
- `pygame_visualizer.py`: Pygame-based visualization of grid worlds and policies
- `main.py`: Terminal-based interface for running the project
- `ai_play.py`: Enhanced visualization and interaction with Pygame
- `large_map_generator.py`: Generation of large-scale environments
- `example_usage.py`: Example code demonstrating environment usage
- `test_*.py`: Test files for different components

## Reinforcement Learning Concepts

This project implements and demonstrates key RL concepts:

- **State**: Represented as (row, col) positions in the grid
- **Action**: Movement directions (UP, RIGHT, DOWN, LEFT)
- **Reward**: Customizable with goal rewards, trap penalties, and step costs
- **Policy**: Mapping from states to actions
- **Value Function**: Estimated future rewards from each state
- **Q-Function**: Estimated future rewards from state-action pairs
- **Exploration vs. Exploitation**: Balanced with epsilon-greedy policies

## Customizing Environments

The `GridWorldEnvironment` class provides rich customization options:

```python
# Create a custom environment
env = GridWorldEnvironment(
    grid=your_grid,                    # 2D array with 0=empty, -100=wall
    start_state=(0, 0),                # Starting position
    goal_states=[(4, 4)],              # Goal positions
    trap_states=[(2, 2)],              # Trap positions
    is_stochastic=True,                # Stochastic transitions
    transition_prob=0.8,               # Probability of intended move
    step_cost=-0.1,                    # Cost for each step
    goal_reward=1.0,                   # Reward for reaching goal
    trap_penalty=1.0,                  # Penalty for falling into trap
    custom_rewards={(1,2,0): 0.5}      # Custom state-action-reward mappings
)
```

## Example Comparisons

The project includes tools for comparing algorithm performance with visualization of:
- Convergence rates
- Policy quality
- Learning efficiency
- Robustness to stochasticity

## Credits

This project was developed as a comprehensive implementation of reinforcement learning algorithms for educational purposes. It draws inspiration from:

- Sutton & Barto's "Reinforcement Learning: An Introduction"
- David Silver's Reinforcement Learning lectures
- OpenAI Gym environments

## License

This project is licensed under the MIT License - see the LICENSE file for details.