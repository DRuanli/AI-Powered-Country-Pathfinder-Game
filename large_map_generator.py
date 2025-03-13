#!/usr/bin/env python3
import numpy as np
import pickle
import time
import os
import argparse
from GridWorldEnvironment import GridWorldEnvironment
from collections import deque

def generate_large_map(size, wall_density=0.2, is_stochastic=False):
    """
    Generate a large random grid world map.
    
    Args:
        size (int): Size of the grid (size x size)
        wall_density (float): Density of walls (0.0 to 1.0)
        is_stochastic (bool): Whether the environment should be stochastic
        
    Returns:
        GridWorldEnvironment: The generated environment
    """
    print(f"Generating {size}x{size} grid world...")
    start_time = time.time()
    
    # Initialize grid
    grid = np.zeros((size, size))
    
    # Set start state (always at top-left corner)
    start_state = (0, 0)
    
    # Set goal state (in the bottom-right quadrant for large maps)
    goal_row = np.random.randint(max(1, size - size//3), size)
    goal_col = np.random.randint(max(1, size - size//3), size)
    goal_states = [(goal_row, goal_col)]
    
    # Set trap states (approx. size/50)
    num_traps = max(1, size // 50)
    print(f"Placing {num_traps} traps...")
    
    trap_states = []
    reserved_positions = [start_state] + goal_states
    
    for _ in range(num_traps):
        attempts = 0
        while attempts < 100:  # Limit attempts to avoid infinite loop
            trap_row = np.random.randint(0, size)
            trap_col = np.random.randint(0, size)
            candidate = (trap_row, trap_col)
            
            if candidate not in reserved_positions:
                trap_states.append(candidate)
                reserved_positions.append(candidate)
                break
            
            attempts += 1
    
    # Add walls
    print("Adding walls...")
    num_walls = int(wall_density * size * size)
    if num_walls > 0:
        # For very large maps, use a more efficient approach
        if size > 100:
            # Create a boolean mask for walls
            wall_mask = np.random.random((size, size)) < wall_density
            
            # Ensure start, goal, and trap positions don't have walls
            for pos in reserved_positions:
                wall_mask[pos] = False
            
            # Apply the mask to the grid
            grid[wall_mask] = -100
        else:
            # For smaller maps, we can use a more precise approach
            walls_added = 0
            max_attempts = num_walls * 2  # Limit attempts
            
            for _ in range(max_attempts):
                if walls_added >= num_walls:
                    break
                    
                wall_row = np.random.randint(0, size)
                wall_col = np.random.randint(0, size)
                candidate = (wall_row, wall_col)
                
                if candidate not in reserved_positions:
                    grid[wall_row, wall_col] = -100  # Set wall value
                    reserved_positions.append(candidate)
                    walls_added += 1
    
    # Ensure path exists (this is crucial)
    print("Ensuring path exists from start to goal...")
    ensure_path_exists(grid, start_state, goal_states[0])
    
    # Create environment
    transition_prob = 0.8 if is_stochastic else 1.0
    env = GridWorldEnvironment(
        grid=grid,
        start_state=start_state,
        goal_states=goal_states,
        trap_states=trap_states,
        is_stochastic=is_stochastic,
        transition_prob=transition_prob,
        step_cost=-0.01,
        goal_reward=1.0,
        trap_penalty=1.0,
        max_steps=float('inf')  # Set to infinity to remove step limit
    )
    
    elapsed_time = time.time() - start_time
    print(f"Grid world generation completed in {elapsed_time:.2f} seconds.")
    
    return env

def ensure_path_exists(grid, start, goal):
    """
    Ensure there's a valid path from start to goal by removing blocking walls.
    Optimized for large grids using breadth-first search.
    
    Args:
        grid (numpy.ndarray): The grid
        start (tuple): Start position (row, col)
        goal (tuple): Goal position (row, col)
    """
    size = grid.shape[0]
    
    # Fast check if path exists
    if has_path(grid, start, goal):
        print("Path already exists!")
        return
    
    print("No path found, creating one...")
    
    # Create path using modified A* (more efficient for large grids)
    visited = set()
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
    
    # Priority queue with manhattan distance heuristic
    # Format: (f_score, current_pos)
    queue = [(manhattan_distance(start, goal), start)]
    
    # Parent dict for path reconstruction
    parent = {start: None}
    
    while queue:
        # Sort by f_score (can use heapq for better performance in very large grids)
        queue.sort(key=lambda x: x[0])
        _, current = queue.pop(0)
        
        if current == goal:
            # Path found, reconstruct it and return
            path = []
            while current != start:
                path.append(current)
                current = parent[current]
            
            # Remove walls along the path
            for pos in path:
                grid[pos] = 0
            
            print(f"Path created with {len(path)} cells")
            return
        
        if current in visited:
            continue
            
        visited.add(current)
        
        # Try each direction
        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc
            next_pos = (nr, nc)
            
            # Check bounds
            if not (0 <= nr < size and 0 <= nc < size):
                continue
                
            # Skip visited positions
            if next_pos in visited:
                continue
                
            # Record parent for path reconstruction
            if next_pos not in parent:
                parent[next_pos] = current
                
            # Add to queue with f_score (g_score + heuristic)
            # Here g_score is implicitly handled by BFS order
            f_score = manhattan_distance(next_pos, goal)
            queue.append((f_score, next_pos))
    
    # If we get here, no path is possible - this shouldn't happen with our grid
    # Fall back to direct line path
    print("Warning: Could not find path with A*. Creating direct path...")
    create_direct_path(grid, start, goal)

def has_path(grid, start, goal):
    """
    Check if there's a path from start to goal using BFS.
    Optimized for large grids.
    
    Args:
        grid (numpy.ndarray): The grid
        start (tuple): Start position (row, col)
        goal (tuple): Goal position (row, col)
        
    Returns:
        bool: True if a path exists, False otherwise
    """
    size = grid.shape[0]
    visited = set([start])
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        
        if current == goal:
            return True
        
        # Try each direction
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = current[0] + dr, current[1] + dc
            next_pos = (nr, nc)
            
            # Check if valid and not a wall
            if (0 <= nr < size and 0 <= nc < size and 
                grid[nr, nc] != -100 and next_pos not in visited):
                visited.add(next_pos)
                queue.append(next_pos)
    
    return False

def create_direct_path(grid, start, goal):
    """
    Create a direct path between start and goal by removing walls.
    
    Args:
        grid (numpy.ndarray): The grid
        start (tuple): Start position (row, col)
        goal (tuple): Goal position (row, col)
    """
    # Get line path
    path = get_line_path(start, goal)
    
    # Remove walls along the path
    for pos in path:
        grid[pos] = 0
    
    print(f"Created direct path with {len(path)} cells")

def get_line_path(start, goal):
    """
    Get a set of points that form a line from start to goal.
    Based on Bresenham's line algorithm.
    
    Args:
        start (tuple): Start position (row, col)
        goal (tuple): Goal position (row, col)
        
    Returns:
        list: List of positions (row, col) forming the path
    """
    x0, y0 = start
    x1, y1 = goal
    path = []
    
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    
    while True:
        path.append((x0, y0))
        
        if x0 == x1 and y0 == y1:
            break
            
        e2 = 2 * err
        if e2 >= dy:
            if x0 == x1:
                break
            err += dy
            x0 += sx
        if e2 <= dx:
            if y0 == y1:
                break
            err += dx
            y0 += sy
    
    return path

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def save_environment(env, filename):
    """
    Save environment to a file.
    
    Args:
        env: GridWorldEnvironment to save
        filename: Filename to save to
    """
    with open(filename, 'wb') as f:
        pickle.dump(env, f)
    print(f"Environment saved to {filename}")

def load_environment(filename):
    """
    Load environment from a file.
    
    Args:
        filename: Filename to load from
        
    Returns:
        GridWorldEnvironment: The loaded environment
    """
    with open(filename, 'rb') as f:
        env = pickle.load(f)
    print(f"Environment loaded from {filename}")
    return env

def main():
    parser = argparse.ArgumentParser(description='Generate large random grid worlds')
    parser.add_argument('--size', type=int, default=100, help='Size of the grid (size x size)')
    parser.add_argument('--walls', type=float, default=0.2, help='Wall density (0.0 to 1.0)')
    parser.add_argument('--stochastic', action='store_true', help='Make environment stochastic')
    parser.add_argument('--output', type=str, default='large_grid.pkl', help='Output file name')
    parser.add_argument('--show', action='store_true', help='Show grid info after generation')
    args = parser.parse_args()
    
    # Validate size
    if args.size < 5 or args.size > 1000:
        print("Size must be between 5 and 1000")
        return
    
    # Validate wall density
    if args.walls < 0.0 or args.walls > 0.9:
        print("Wall density must be between 0.0 and 0.9")
        return
    
    # Generate environment
    env = generate_large_map(args.size, args.walls, args.stochastic)
    
    # Show grid info if requested
    if args.show:
        # Print grid info
        print(f"\nGrid Size: {args.size}x{args.size}")
        print(f"Start Position: {env.start_state}")
        print(f"Goal Position: {env.goal_states[0]}")
        print(f"Number of Traps: {len(env.trap_states)}")
        
        # For very large grids, don't try to print the whole thing
        if args.size <= 50:
            env.render()
    
    # Save environment
    save_environment(env, args.output)

if __name__ == "__main__":
    main()