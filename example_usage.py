import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from GridWorldEnvironment import GridWorldEnvironment

def create_standard_grid():
    """Create a standard 5x5 grid with walls, goal, and trap."""
    grid = np.zeros((5, 5))
    
    # Add walls
    walls = [(1, 1), (1, 3), (3, 1), (3, 3)]
    for r, c in walls:
        grid[r, c] = -1
    
    return grid

def create_complex_grid(size=10, reserved_positions=None):
    """Create a more complex grid with multiple goals and traps.
    
    Args:
        size: Size of the grid
        reserved_positions: List of (row, col) positions that should not be walls
    """
    if reserved_positions is None:
        reserved_positions = []
    
    grid = np.zeros((size, size))
    
    # Add walls in a maze-like pattern
    for i in range(2, size, 2):
        for j in range(size):
            if j != i % size and (i, j) not in reserved_positions:  # Leave openings
                grid[i, j] = -1
    
    # Add some vertical walls too
    for i in range(size):
        for j in range(3, size, 3):
            if i != j % size and grid[i, j] != -1 and (i, j) not in reserved_positions:
                grid[i, j] = -1
    
    return grid

def demonstrate_basic_usage():
    """Demonstrate basic usage of the GridWorldEnvironment."""
    print("Demonstrating basic environment usage...")
    
    # Create a standard grid
    grid = create_standard_grid()
    
    # Define start, goal, and trap states
    start_state = (0, 0)
    goal_states = [(4, 4)]
    trap_states = [(2, 2)]
    
    # Create environment
    env = GridWorldEnvironment(
        grid=grid,
        start_state=start_state,
        goal_states=goal_states,
        trap_states=trap_states,
        step_cost=-0.1
    )
    
    # Display the initial environment
    print("Initial environment:")
    env.render()
    
    # Take some manual steps
    actions = [env.RIGHT, env.DOWN, env.RIGHT, env.DOWN, env.RIGHT]
    print("\nTaking steps:")
    
    for i, action in enumerate(actions):
        next_state, reward, done, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Next state={next_state}, Reward={reward}, Done={done}")
    
    # Show environment after steps
    print("\nEnvironment after steps:")
    env.render(agent_history=True)
    
    # Run a complete episode with random actions
    env.reset()
    print("\nRunning a random episode:")
    total_reward, steps, states = env.run_episode()
    print(f"Episode completed in {steps} steps with total reward: {total_reward}")
    
    # Show the final state with history
    print("\nFinal state with agent history:")
    env.render(agent_history=True)

def demonstrate_stochastic_transitions():
    """Demonstrate stochastic transitions in the environment."""
    print("Demonstrating stochastic transitions...")
    
    grid = create_standard_grid()
    start_state = (0, 0)
    goal_states = [(4, 4)]
    trap_states = [(2, 2)]
    
    # Create a stochastic environment
    env = GridWorldEnvironment(
        grid=grid,
        start_state=start_state,
        goal_states=goal_states,
        trap_states=trap_states,
        is_stochastic=True,
        transition_prob=0.8,
        step_cost=-0.1
    )
    
    # Run a sample of steps with the same action and observe different outcomes
    action = env.RIGHT
    print(f"\nAttempting to move RIGHT from start state 10 times:")
    
    for i in range(10):
        env.reset()
        next_state, reward, done, _ = env.step(action)
        print(f"Attempt {i+1}: Intended action=RIGHT, Result={next_state}")
    
    # Run a full episode in the stochastic environment
    env.reset()
    print("\nRunning a full episode with stochastic transitions:")
    total_reward, steps, states = env.run_episode()
    print(f"Episode completed in {steps} steps with total reward: {total_reward}")
    
    # Show the trajectory
    print("\nStochastic trajectory:")
    env.render(agent_history=True)

def demonstrate_complex_environment():
    """Demonstrate a more complex environment with multiple goals and traps."""
    print("Demonstrating complex environment...")
    
    # Define multiple start, goal, and trap states
    start_state = (0, 0)
    goal_states = [(9, 9), (0, 9), (9, 0)]
    trap_states = [(4, 4), (5, 5), (6, 6)]
    
    # Create a complex grid, reserving positions for start, goals, and traps
    reserved_positions = [start_state] + goal_states + trap_states
    grid = create_complex_grid(10, reserved_positions)
    
    # Add custom rewards
    custom_rewards = {
        (2, 7, GridWorldEnvironment.UP): 0.5,
        (7, 2, GridWorldEnvironment.RIGHT): 0.3
    }
    
    # Create the environment
    env = GridWorldEnvironment(
        grid=grid,
        start_state=start_state,
        goal_states=goal_states,
        trap_states=trap_states,
        is_stochastic=True,
        transition_prob=0.8,
        step_cost=-0.01,
        goal_reward=1.0,
        trap_penalty=-1.0,
        custom_rewards=custom_rewards
    )
    
    # Visualize the initial environment
    print("\nComplex environment:")
    env.render()
    
    # Run an episode
    print("\nRunning an episode in the complex environment:")
    total_reward, steps, states = env.run_episode()
    print(f"Episode completed in {steps} steps with total reward: {total_reward}")
    
    # Animate the episode
    print("\nAnimating the episode (you'll need to be in a Jupyter notebook to see this):")
    animation = env.animate_episode(interval=100)
    display(animation)

def demonstrate_analysis_tools():
    """Demonstrate analysis tools for reinforcement learning."""
    print("Demonstrating analysis tools...")
    
    grid = create_standard_grid()
    start_state = (0, 0)
    goal_states = [(4, 4)]
    trap_states = [(2, 2)]
    
    env = GridWorldEnvironment(
        grid=grid,
        start_state=start_state,
        goal_states=goal_states,
        trap_states=trap_states,
        step_cost=-0.1
    )
    
    # Create a sample value function to visualize
    value_function = np.zeros_like(grid)
    for r in range(5):
        for c in range(5):
            if grid[r, c] != -1:  # Not a wall
                # Simple distance-based value
                distance = abs(r - goal_states[0][0]) + abs(c - goal_states[0][1])
                value_function[r, c] = 1.0 / (distance + 1)
                
                # Decrease value near traps
                for trap_r, trap_c in trap_states:
                    trap_distance = abs(r - trap_r) + abs(c - trap_c)
                    if trap_distance <= 1:
                        value_function[r, c] *= 0.5
    
    # Create a sample policy to visualize
    policy = np.zeros_like(grid, dtype=int)
    for r in range(5):
        for c in range(5):
            if (r, c) in goal_states or (r, c) in trap_states or grid[r, c] == -1:
                continue
                
            # Simple policy: move toward the goal
            if r < goal_states[0][0]:
                policy[r, c] = env.DOWN
            elif r > goal_states[0][0]:
                policy[r, c] = env.UP
            elif c < goal_states[0][1]:
                policy[r, c] = env.RIGHT
            else:
                policy[r, c] = env.LEFT
    
    # Visualize value function
    print("\nValue function visualization:")
    env.render(show_values=True, values=value_function)
    
    # Visualize policy
    print("\nPolicy visualization:")
    env.render(policy=policy)
    
    # Visualize both
    print("\nCombined visualization (value function and policy):")
    env.render(show_values=True, values=value_function, policy=policy)

def main():
    """Run all demonstrations."""
    demonstrate_basic_usage()
    print("\n" + "-"*50 + "\n")
    demonstrate_stochastic_transitions()
    print("\n" + "-"*50 + "\n")
    demonstrate_complex_environment()
    print("\n" + "-"*50 + "\n")
    demonstrate_analysis_tools()

if __name__ == "__main__":
    main()