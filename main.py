#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from GridWorldEnvironment import GridWorldEnvironment
from dynamic_programming import DynamicProgramming
from td_learning import TDLearning
from deep_rl import dqn
import time

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the game header."""
    print("\n" + "=" * 50)
    print("  REINFORCEMENT LEARNING GRID WORLD GAME")
    print("=" * 50 + "\n")

def create_custom_grid():
    """Allow user to create a custom grid world."""
    print("Create your custom grid world!\n")
    
    # Get grid dimensions
    while True:
        try:
            height = int(input("Enter grid height (2-10): "))
            width = int(input("Enter grid width (2-10): "))
            if 2 <= height <= 10 and 2 <= width <= 10:
                break
            else:
                print("Dimensions must be between 2 and 10.")
        except ValueError:
            print("Please enter valid numbers.")
    
    # Initialize grid with empty cells
    grid = np.zeros((height, width))
    
    # Get start position
    while True:
        try:
            print("\nSet start position:")
            start_row = int(input(f"Row (0-{height-1}): "))
            start_col = int(input(f"Column (0-{width-1}): "))
            if 0 <= start_row < height and 0 <= start_col < width:
                start_state = (start_row, start_col)
                break
            else:
                print("Position out of bounds.")
        except ValueError:
            print("Please enter valid numbers.")
    
    # Get goal positions
    goal_states = []
    while True:
        try:
            print("\nAdd goal position (enter -1 to finish):")
            goal_row = int(input(f"Row (0-{height-1}, -1 to finish): "))
            if goal_row == -1:
                if not goal_states:
                    print("Please add at least one goal.")
                    continue
                break
            goal_col = int(input(f"Column (0-{width-1}): "))
            if 0 <= goal_row < height and 0 <= goal_col < width:
                if (goal_row, goal_col) == start_state:
                    print("Cannot set start position as goal.")
                    continue
                if (goal_row, goal_col) in goal_states:
                    print("Position already set as goal.")
                    continue
                goal_states.append((goal_row, goal_col))
                print(f"Goal added at ({goal_row}, {goal_col})")
            else:
                print("Position out of bounds.")
        except ValueError:
            print("Please enter valid numbers.")
    
    # Get trap positions
    trap_states = []
    while True:
        try:
            print("\nAdd trap position (enter -1 to finish):")
            trap_row = int(input(f"Row (0-{height-1}, -1 to finish): "))
            if trap_row == -1:
                break
            trap_col = int(input(f"Column (0-{width-1}): "))
            if 0 <= trap_row < height and 0 <= trap_col < width:
                if (trap_row, trap_col) == start_state:
                    print("Cannot set start position as trap.")
                    continue
                if (trap_row, trap_col) in goal_states:
                    print("Cannot set goal position as trap.")
                    continue
                if (trap_row, trap_col) in trap_states:
                    print("Position already set as trap.")
                    continue
                trap_states.append((trap_row, trap_col))
                print(f"Trap added at ({trap_row}, {trap_col})")
            else:
                print("Position out of bounds.")
        except ValueError:
            print("Please enter valid numbers.")
    
    # Get wall positions
    walls = []
    while True:
        try:
            print("\nAdd wall position (enter -1 to finish):")
            wall_row = int(input(f"Row (0-{height-1}, -1 to finish): "))
            if wall_row == -1:
                break
            wall_col = int(input(f"Column (0-{width-1}): "))
            if 0 <= wall_row < height and 0 <= wall_col < width:
                if (wall_row, wall_col) == start_state:
                    print("Cannot set start position as wall.")
                    continue
                if (wall_row, wall_col) in goal_states:
                    print("Cannot set goal position as wall.")
                    continue
                if (wall_row, wall_col) in trap_states:
                    print("Cannot set trap position as wall.")
                    continue
                walls.append((wall_row, wall_col))
                grid[wall_row, wall_col] = -100  # Set wall value
                print(f"Wall added at ({wall_row}, {wall_col})")
            else:
                print("Position out of bounds.")
        except ValueError:
            print("Please enter valid numbers.")
    
    # Get environment parameters
    is_stochastic = input("\nMake environment stochastic? (y/n): ").lower() == 'y'
    transition_prob = 0.8
    if is_stochastic:
        while True:
            try:
                transition_prob = float(input("Enter transition probability (0.5-1.0): "))
                if 0.5 <= transition_prob <= 1.0:
                    break
                else:
                    print("Probability must be between 0.5 and 1.0.")
            except ValueError:
                print("Please enter a valid number.")
    
    step_cost = -0.1
    while True:
        try:
            step_cost = float(input("\nEnter step cost (negative value, e.g. -0.1): "))
            if step_cost <= 0:
                break
            else:
                print("Step cost must be negative or zero.")
        except ValueError:
            print("Please enter a valid number.")
    
    goal_reward = 1.0
    while True:
        try:
            goal_reward = float(input("\nEnter goal reward (positive value, e.g. 1.0): "))
            if goal_reward > 0:
                break
            else:
                print("Goal reward must be positive.")
        except ValueError:
            print("Please enter a valid number.")
    
    trap_penalty = 1.0
    while True:
        try:
            trap_penalty = float(input("\nEnter trap penalty (positive value, e.g. 1.0): "))
            if trap_penalty > 0:
                break
            else:
                print("Trap penalty must be positive.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Create and return the environment
    env = GridWorldEnvironment(
        grid=grid,
        start_state=start_state,
        goal_states=goal_states,
        trap_states=trap_states,
        is_stochastic=is_stochastic,
        transition_prob=transition_prob,
        step_cost=step_cost,
        goal_reward=goal_reward,
        trap_penalty=trap_penalty
    )
    
    return env

def load_predefined_grid():
    """Load a predefined grid world."""
    print("Select a predefined grid world:\n")
    print("1. Simple 5x5 Grid")
    print("2. Complex 10x10 Grid")
    print("3. Maze 8x8 Grid")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == "1":
        # Simple 5x5 grid
        grid = np.zeros((5, 5))
        walls = [(1, 1), (1, 3), (3, 1), (3, 3)]
        for r, c in walls:
            grid[r, c] = -100
        
        start_state = (0, 0)
        goal_states = [(4, 4)]
        trap_states = [(2, 2)]
        
        env = GridWorldEnvironment(
            grid=grid,
            start_state=start_state,
            goal_states=goal_states,
            trap_states=trap_states,
            is_stochastic=False,
            step_cost=-0.1,
            goal_reward=1.0,
            trap_penalty=1.0
        )
        
    elif choice == "2":
        # Complex 10x10 grid
        grid = np.zeros((10, 10))
        
        # Add walls in a maze-like pattern
        for i in range(2, 10, 2):
            for j in range(10):
                if j != i % 10:
                    grid[i, j] = -100
        
        # Add some vertical walls too
        for i in range(10):
            for j in range(3, 10, 3):
                if i != j % 10 and grid[i, j] != -100:
                    grid[i, j] = -100
        
        start_state = (0, 0)
        goal_states = [(9, 9)]
        trap_states = [(5, 5), (7, 3)]
        
        env = GridWorldEnvironment(
            grid=grid,
            start_state=start_state,
            goal_states=goal_states,
            trap_states=trap_states,
            is_stochastic=True,
            transition_prob=0.8,
            step_cost=-0.05,
            goal_reward=1.0,
            trap_penalty=1.0
        )
        
    elif choice == "3":
        # Maze 8x8 grid
        grid = np.zeros((8, 8))
        
        # Create a maze pattern
        walls = [
            (0, 2), (0, 5), (1, 2), (1, 5), (1, 7),
            (2, 0), (2, 2), (2, 3), (2, 5), (3, 3),
            (3, 5), (3, 6), (3, 7), (4, 1), (4, 5),
            (5, 1), (5, 2), (5, 3), (5, 5), (5, 7),
            (6, 5), (7, 1), (7, 3)
        ]
        
        for r, c in walls:
            grid[r, c] = -100
        
        start_state = (0, 0)
        goal_states = [(7, 7)]
        trap_states = [(2, 6), (4, 3), (6, 2)]
        
        env = GridWorldEnvironment(
            grid=grid,
            start_state=start_state,
            goal_states=goal_states,
            trap_states=trap_states,
            is_stochastic=False,
            step_cost=-0.1,
            goal_reward=1.0,
            trap_penalty=1.0
        )
    
    else:
        print("Invalid choice. Loading simple grid.")
        return load_predefined_grid()
    
    return env

def choose_agent(env):
    """Choose an RL agent to guide you."""
    print("\nChoose an RL agent to help you (it will show you the optimal policy):\n")
    print("1. Value Iteration")
    print("2. Policy Iteration")
    print("3. Q-Learning")
    print("4. SARSA")
    print("5. Deep Q-Network (DQN)")
    print("6. No agent (play on your own)")
    
    choice = input("\nEnter your choice (1-6): ")
    
    if choice == "1":
        print("\nTraining Value Iteration agent...")
        dp = DynamicProgramming(env)
        policy, values, _ = dp.value_iteration()
        env.render(show_values=True, values=values, policy=policy)
        print("\nValue Iteration agent trained!")
        return policy
        
    elif choice == "2":
        print("\nTraining Policy Iteration agent...")
        dp = DynamicProgramming(env)
        policy, values, _ = dp.policy_iteration()
        env.render(show_values=True, values=values, policy=policy)
        print("\nPolicy Iteration agent trained!")
        return policy
        
    elif choice == "3":
        print("\nTraining Q-Learning agent (this might take a moment)...")
        td = TDLearning(env)
        _, policy, _ = td.q_learning(episodes=500)
        env.render(agent_history=False)
        print("\nQ-Learning agent trained!")
        return policy
        
    elif choice == "4":
        print("\nTraining SARSA agent (this might take a moment)...")
        td = TDLearning(env)
        _, policy, _ = td.sarsa(episodes=500)
        env.render(agent_history=False)
        print("\nSARSA agent trained!")
        return policy
        
    elif choice == "5":
        print("\nTraining DQN agent (this might take a moment)...")
        state_size = env.height * env.width
        agent, policy, _ = dqn(env, episodes=300)
        env.render(agent_history=False)
        print("\nDQN agent trained!")
        return policy
        
    else:
        print("\nYou'll play on your own!")
        return None

def play_game(env, policy=None):
    """Play the grid world game."""
    state = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    action_map = {
        "w": env.UP,
        "a": env.LEFT,
        "s": env.DOWN,
        "d": env.RIGHT
    }
    
    action_names = {
        env.UP: "UP",
        env.RIGHT: "RIGHT",
        env.DOWN: "DOWN",
        env.LEFT: "LEFT"
    }
    
    print("\nStarting the game!")
    print("Use WASD keys to move (w=up, a=left, s=down, d=right)")
    print("Press 'q' to quit at any time.")
    print("\nInitial state:")
    env.render()
    
    while not done:
        # Show optimal action if policy is available
        if policy is not None:
            if isinstance(policy, dict):
                if state in policy:
                    suggested_action = policy[state]
                    print(f"Suggested move: {action_names[suggested_action]}")
                else:
                    print("No suggested move for this state.")
            else:
                # Policy is a numpy array
                r, c = state
                if 0 <= r < policy.shape[0] and 0 <= c < policy.shape[1]:
                    suggested_action = policy[r, c]
                    print(f"Suggested move: {action_names[suggested_action]}")
                else:
                    print("No suggested move for this state.")
        
        # Get user action
        while True:
            action_key = input("\nEnter your move (w/a/s/d, q to quit): ").lower()
            if action_key == 'q':
                print("\nQuitting game...")
                return
            if action_key in action_map:
                action = action_map[action_key]
                break
            else:
                print("Invalid move. Use w/a/s/d keys.")
        
        # Take the action
        next_state, reward, done, info = env.step(action)
        
        # Update state and display
        state = next_state
        total_reward += reward
        steps += 1
        
        clear_screen()
        print_header()
        print(f"Move: {action_names[action]}")
        print(f"Reward: {reward:.2f}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Steps: {steps}")
        
        if info.get('is_goal', False):
            print("\nCongratulations! You reached the goal!")
        elif info.get('is_trap', False):
            print("\nOh no! You fell into a trap!")
        elif info.get('is_max_steps', False):
            print("\nYou reached the maximum number of steps.")
        
        env.render(agent_history=True)
        
        # Add a slight delay for better user experience
        time.sleep(0.5)
    
    print("\nGame over!")
    print(f"Final reward: {total_reward:.2f}")
    print(f"Steps taken: {steps}")
    
    play_again = input("\nPlay again? (y/n): ").lower() == 'y'
    return play_again

def main():
    parser = argparse.ArgumentParser(description='Reinforcement Learning Grid World Game')
    parser.add_argument('--custom', action='store_true', help='Create a custom grid')
    parser.add_argument('--predefined', action='store_true', help='Use a predefined grid')
    args = parser.parse_args()
    
    playing = True
    while playing:
        clear_screen()
        print_header()
        
        # Choose grid
        if args.custom:
            env = create_custom_grid()
        elif args.predefined:
            env = load_predefined_grid()
        else:
            print("Select grid mode:\n")
            print("1. Create custom grid")
            print("2. Use predefined grid")
            
            choice = input("\nEnter your choice (1-2): ")
            if choice == "1":
                env = create_custom_grid()
            else:
                env = load_predefined_grid()
        
        # Choose agent
        policy = choose_agent(env)
        
        # Play the game
        clear_screen()
        print_header()
        playing = play_game(env, policy)

if __name__ == "__main__":
    main()