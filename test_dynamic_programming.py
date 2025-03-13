import numpy as np
import matplotlib.pyplot as plt
from GridWorldEnvironment import GridWorldEnvironment
from dynamic_programming import DynamicProgramming

def create_test_environment(is_stochastic=False):
    """
    Create a test environment for the dynamic programming algorithms.
    
    Args:
        is_stochastic (bool): Whether the environment should have stochastic transitions
        
    Returns:
        GridWorldEnvironment: The test environment
    """
    # Create a 5x5 grid with walls
    grid = np.zeros((5, 5))
    
    # Add walls
    walls = [(1, 1), (1, 3), (3, 1), (3, 3)]
    for r, c in walls:
        grid[r, c] = -100  # Using WALL value
    
    # Define start state
    start_state = (0, 0)
    
    # Define goal and trap states
    goal_states = [(4, 4)]
    trap_states = [(2, 2)]
    
    # Create environment
    env = GridWorldEnvironment(
        grid=grid,
        start_state=start_state,
        goal_states=goal_states,
        trap_states=trap_states,
        is_stochastic=is_stochastic,
        transition_prob=0.8 if is_stochastic else 1.0,
        step_cost=-0.1,
        goal_reward=1.0,
        trap_penalty=1.0
    )
    
    return env

def test_value_iteration(environment, gamma=0.9, epsilon=1e-6):
    """
    Test the Value Iteration algorithm.
    
    Args:
        environment: The GridWorldEnvironment
        gamma (float): Discount factor
        epsilon (float): Convergence threshold
    """
    print("Testing Value Iteration...")
    
    # Create dynamic programming instance
    dp = DynamicProgramming(environment)
    
    # Run value iteration with increased convergence threshold for stochastic environments
    epsilon_adjusted = epsilon if not environment.is_stochastic else 1e-4
    policy, values, metrics = dp.value_iteration(gamma=gamma, epsilon=epsilon_adjusted, max_iterations=300)
    
    # Visualize results
    print("\nValue Function:")
    dp.visualize_value_function(values, title="Value Iteration - Value Function")
    
    print("\nPolicy:")
    dp.visualize_policy(policy, value_function=values, title="Value Iteration - Policy")
    
    print("\nConvergence:")
    dp.visualize_convergence(metrics, algorithm="Value Iteration")
    
    # Convert policy from 2D array to dictionary format
    policy_dict = {}
    for r in range(environment.height):
        for c in range(environment.width):
            if environment.grid[r, c] != environment.WALL and (r, c) not in environment.goal_states and (r, c) not in environment.trap_states:
                policy_dict[(r, c)] = policy[r, c]
    
    # Run a test episode with the policy
    total_reward, steps, states = environment.run_episode(policy=policy_dict)
    print(f"\nTest Episode Results:")
    print(f"Total reward: {total_reward}")
    print(f"Steps taken: {steps}")
    
    # Visualize the episode
    environment.render(agent_history=True)
    
    return policy, values, metrics

def test_policy_iteration(environment, gamma=0.9):
    """
    Test the Policy Iteration algorithm.
    
    Args:
        environment: The GridWorldEnvironment
        gamma (float): Discount factor
    """
    print("Testing Policy Iteration...")
    
    # Create dynamic programming instance
    dp = DynamicProgramming(environment)
    
    # Run policy iteration with adjusted parameters for stochastic environments
    eval_epsilon = 1e-6 if not environment.is_stochastic else 1e-4
    policy, values, metrics = dp.policy_iteration(gamma=gamma, max_iterations=30, eval_epsilon=eval_epsilon)
    
    # Visualize results
    print("\nValue Function:")
    dp.visualize_value_function(values, title="Policy Iteration - Value Function")
    
    print("\nPolicy:")
    dp.visualize_policy(policy, value_function=values, title="Policy Iteration - Policy")
    
    print("\nConvergence:")
    dp.visualize_convergence(metrics, algorithm="Policy Iteration")
    
    # Convert policy from 2D array to dictionary format
    policy_dict = {}
    for r in range(environment.height):
        for c in range(environment.width):
            if environment.grid[r, c] != environment.WALL and (r, c) not in environment.goal_states and (r, c) not in environment.trap_states:
                policy_dict[(r, c)] = policy[r, c]
    
    # Run a test episode with the policy
    total_reward, steps, states = environment.run_episode(policy=policy_dict)
    print(f"\nTest Episode Results:")
    print(f"Total reward: {total_reward}")
    print(f"Steps taken: {steps}")
    
    # Visualize the episode
    environment.render(agent_history=True)
    
    return policy, values, metrics

def compare_algorithms(environment):
    """
    Compare Value Iteration and Policy Iteration algorithms.
    
    Args:
        environment: The GridWorldEnvironment
    """
    print("Comparing Value Iteration and Policy Iteration...")
    
    # Create dynamic programming instance
    dp = DynamicProgramming(environment)
    
    # Run both algorithms
    vi_policy, vi_values, vi_metrics = dp.value_iteration()
    pi_policy, pi_values, pi_metrics = dp.policy_iteration()
    
    # Compare performance
    dp.compare_algorithms(vi_metrics, pi_metrics)
    
    # Compare policies
    print("\nPolicy Comparison:")
    print("Value Iteration and Policy Iteration policies match:", np.array_equal(vi_policy, pi_policy))
    
    # Compare value functions
    value_diff = np.abs(vi_values - pi_values)
    print("Value Function Difference:")
    print(f"Max difference: {np.nanmax(value_diff)}")
    print(f"Mean difference: {np.nanmean(value_diff)}")

def test_stochastic_environment():
    """Test both algorithms in a stochastic environment."""
    print("Testing in a stochastic environment...\n")
    
    # Create a stochastic environment
    env = create_test_environment(is_stochastic=True)
    
    # Display environment
    print("Environment:")
    env.render()
    
    # Test both algorithms
    vi_results = test_value_iteration(env)
    print("\n" + "="*50 + "\n")
    pi_results = test_policy_iteration(env)
    print("\n" + "="*50 + "\n")
    
    # Compare algorithms
    compare_algorithms(env)

def test_deterministic_environment():
    """Test both algorithms in a deterministic environment."""
    print("Testing in a deterministic environment...\n")
    
    # Create a deterministic environment
    env = create_test_environment(is_stochastic=False)
    
    # Display environment
    print("Environment:")
    env.render()
    
    # Test both algorithms
    vi_results = test_value_iteration(env)
    print("\n" + "="*50 + "\n")
    pi_results = test_policy_iteration(env)
    print("\n" + "="*50 + "\n")
    
    # Compare algorithms
    compare_algorithms(env)

def main():
    """Run all tests."""
    # Test in deterministic environment
    test_deterministic_environment()
    
    print("\n" + "="*70 + "\n")
    
    # Test in stochastic environment
    test_stochastic_environment()

if __name__ == "__main__":
    main()