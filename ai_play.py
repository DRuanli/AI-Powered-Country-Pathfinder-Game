#!/usr/bin/env python3
import numpy as np
import argparse
import os
import pickle
import sys
from GridWorldEnvironment import GridWorldEnvironment
from dynamic_programming import DynamicProgramming
from td_learning import TDLearning
from deep_rl import dqn
from pygame_visualizer import visualize_with_ai, visualize_interactive, LargeMapVisualizer

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the game header."""
    print("\n" + "=" * 50)
    print("  REINFORCEMENT LEARNING GRID WORLD GAME")
    print("=" * 50 + "\n")

def create_custom_grid():
    """Automatically create a custom grid world with minimal user input."""
    print("Creating custom grid world!\n")
    
    # Ask for grid size
    while True:
        try:
            size = int(input("Enter grid size (5-15): "))
            if 5 <= size <= 15:
                break
            else:
                print("Size must be between 5 and 15.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Ask for difficulty level
    while True:
        try:
            print("\nSelect difficulty level:")
            print("1. Easy (few walls and traps)")
            print("2. Medium (moderate walls and traps)")
            print("3. Hard (many walls and traps)")
            difficulty = int(input("\nEnter difficulty (1-3): "))
            if 1 <= difficulty <= 3:
                break
            else:
                print("Please select 1, 2, or 3.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Initialize grid
    grid = np.zeros((size, size))
    
    # Set wall density based on difficulty
    if difficulty == 1:
        wall_density = 0.1
        num_traps = max(1, size // 5)
    elif difficulty == 2:
        wall_density = 0.2
        num_traps = max(2, size // 4)
    else:
        wall_density = 0.3
        num_traps = max(3, size // 3)
    
    # Set start state (always at top-left)
    start_state = (0, 0)
    
    # Set goal state (usually at bottom-right but with some randomness)
    goal_row = np.random.randint(size//2, size)
    goal_col = np.random.randint(size//2, size)
    goal_states = [(goal_row, goal_col)]
    
    # Generate traps
    trap_states = []
    for _ in range(num_traps):
        while True:
            trap_row = np.random.randint(1, size-1)
            trap_col = np.random.randint(1, size-1)
            candidate = (trap_row, trap_col)
            
            # Make sure traps don't overlap with start, goals, or other traps
            if (candidate != start_state and 
                candidate not in goal_states and 
                candidate not in trap_states):
                trap_states.append(candidate)
                break
    
    # Generate walls
    reserved_positions = [start_state] + goal_states + trap_states
    num_walls = int(wall_density * size * size)
    for _ in range(num_walls):
        attempts = 0
        while attempts < 10:  # Limit attempts to avoid infinite loop
            wall_row = np.random.randint(0, size)
            wall_col = np.random.randint(0, size)
            candidate = (wall_row, wall_col)
            
            if candidate not in reserved_positions:
                grid[wall_row, wall_col] = -100  # Set wall value
                reserved_positions.append(candidate)
                break
            
            attempts += 1
    
    # Ensure a path exists from start to goal
    ensure_path_exists(grid, start_state, goal_states[0])
    
    # Set environment parameters based on difficulty
    is_stochastic = difficulty > 1
    transition_prob = 0.8
    step_cost = -0.1
    goal_reward = 1.0
    trap_penalty = 1.0
    
    # Create and display the environment
    env = GridWorldEnvironment(
        grid=grid,
        start_state=start_state,
        goal_states=goal_states,
        trap_states=trap_states,
        is_stochastic=is_stochastic,
        transition_prob=transition_prob,
        step_cost=step_cost,
        goal_reward=goal_reward,
        trap_penalty=trap_penalty,
        max_steps=float('inf')  # Remove step limit
    )
    
    # Show the generated grid
    print("\nGenerated grid world:")
    env.render()
    
    print(f"\nStart position: {start_state}")
    print(f"Goal position(s): {goal_states}")
    print(f"Trap position(s): {trap_states}")
    print(f"Environment is {'stochastic' if is_stochastic else 'deterministic'}")
    print(f"Step cost: {step_cost}, Goal reward: {goal_reward}, Trap penalty: {trap_penalty}")
    print("No maximum step limit")
    
    input("\nPress Enter to continue...")
    
    return env

def ensure_path_exists(grid, start, goal):
    """Ensure there's a valid path from start to goal by removing blocking walls."""
    # Simple implementation using A* algorithm
    size = grid.shape[0]
    visited = set()
    queue = [(manhattan_distance(start, goal), 0, start, [])]  # (f, g, position, path)
    
    while queue:
        # Sort by f value (A* heuristic)
        queue.sort()
        _, g, current, path = queue.pop(0)
        
        if current == goal:
            # Path found, no need to change the grid
            return
        
        if current in visited:
            continue
        
        visited.add(current)
        
        # Try each direction
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            next_pos = (nx, ny)
            
            # Check if within bounds
            if 0 <= nx < size and 0 <= ny < size:
                if next_pos not in visited:
                    # If there's a wall, remove it to create a path
                    if grid[nx, ny] == -100:
                        grid[nx, ny] = 0  # Remove wall
                    
                    # Add to queue
                    new_g = g + 1
                    new_f = new_g + manhattan_distance(next_pos, goal)
                    new_path = path + [next_pos]
                    queue.append((new_f, new_g, next_pos, new_path))

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

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
            trap_penalty=1.0,
            max_steps=float('inf')  # Remove step limit
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
            trap_penalty=1.0,
            max_steps=float('inf')  # Remove step limit
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
            trap_penalty=1.0,
            max_steps=float('inf')  # Remove step limit
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
        agent, policy, _ = dqn(env, episodes=300)
        env.render(agent_history=False)
        print("\nDQN agent trained!")
        return policy
        
    else:
        print("\nYou'll play on your own!")
        return None

def load_large_map(filename):
    """Load a large map from a pickle file."""
    try:
        with open(filename, 'rb') as f:
            env = pickle.load(f)
            
        # Remove max steps limitation
        env.max_steps = float('inf')
            
        print(f"Successfully loaded map from {filename}")
        print(f"Grid dimensions: {env.height}x{env.width}")
        print(f"Start position: {env.start_state}")
        print(f"Goal position(s): {env.goal_states}")
        print(f"Number of traps: {len(env.trap_states)}")
        print("Max steps limitation removed")
        return env
    except Exception as e:
        print(f"Error loading map from {filename}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Reinforcement Learning Grid World Game with Pygame Visualization')
    parser.add_argument('--custom', action='store_true', help='Create a custom grid')
    parser.add_argument('--predefined', action='store_true', help='Use a predefined grid')
    parser.add_argument('--interactive', action='store_true', help='Play the game yourself')
    parser.add_argument('--load', type=str, help='Load map from pickle file')
    parser.add_argument('--no-train', action='store_true', help='Skip training an agent (for large maps)')
    args = parser.parse_args()
    
    clear_screen()
    print_header()
    
    # Load map from file if specified
    if args.load:
        env = load_large_map(args.load)
        if env is None:
            print("Failed to load map. Exiting.")
            return
    # Otherwise choose grid
    elif args.custom:
        env = create_custom_grid()
    elif args.predefined:
        env = load_predefined_grid()
    else:
        print("Select grid mode:\n")
        print("1. Create custom grid")
        print("2. Use predefined grid")
        print("3. Load from file")
        
        choice = input("\nEnter your choice (1-3): ")
        if choice == "1":
            env = create_custom_grid()
        elif choice == "2":
            env = load_predefined_grid()
        elif choice == "3":
            filename = input("Enter the path to the map file: ")
            env = load_large_map(filename)
            if env is None:
                print("Failed to load map. Exiting.")
                return
        else:
            print("Invalid choice. Exiting.")
            return
    
    # Detect if map is very large
    is_large_map = env.height > 50 or env.width > 50
    
    # Check if the user wants to play interactively
    if args.interactive:
        print("\nStarting interactive game...")
        if is_large_map:
            # Use the large map visualizer for interactive mode
            visualizer = LargeMapVisualizer(env)
            visualizer.run_interactive()
        else:
            visualize_interactive(env)
        return
    
    # Choose agent, but skip for large maps if requested
    policy = None
    if not args.no_train and not is_large_map:
        policy = choose_agent(env)
    elif is_large_map:
        print("\nLarge map detected!")
        if not args.no_train:
            print("Warning: Training agents on large maps can be very slow.")
            train_anyway = input("Do you want to train an agent anyway? (y/n): ").lower()
            if train_anyway == 'y':
                policy = choose_agent(env)
            else:
                print("Skipping agent training.")
        else:
            print("Skipping agent training as requested.")
    
    # Start visualization
    print("\nStarting visualization...")
    print("Press SPACE to play/pause, +/- to adjust speed, R to reset, Q to quit")
    
    if is_large_map:
        print("Large map controls: WASD to move camera, Mouse wheel to zoom")
        visualizer = LargeMapVisualizer(env)
        if policy:
            visualizer.set_policy(policy)
        visualizer.run_episode(policy)
    else:
        # Calculate an appropriate cell size based on grid dimensions
        max_dim = max(env.height, env.width)
        if max_dim <= 5:
            cell_size = 80
        elif max_dim <= 10:
            cell_size = 60
        else:
            cell_size = 40
        
        # Run visualization
        visualize_with_ai(env, policy, cell_size=cell_size)

if __name__ == "__main__":
    main()