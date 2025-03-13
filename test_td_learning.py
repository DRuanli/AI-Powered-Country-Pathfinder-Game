import numpy as np
import matplotlib.pyplot as plt
from GridWorldEnvironment import GridWorldEnvironment
from td_learning import TDLearning

def create_test_environment(is_stochastic=False):
    """
    Create a test environment for the TD learning algorithms.
    
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

def test_q_learning(environment, episodes=500, title_prefix="Q-Learning"):
    """
    Test the Q-Learning algorithm.
    
    Args:
        environment: The GridWorldEnvironment
        episodes (int): Number of episodes to run
        title_prefix (str): Prefix for plot titles
    """
    print(f"Testing {title_prefix}...")
    
    # Create TD Learning instance
    td = TDLearning(environment)
    
    # Run Q-Learning with decaying epsilon
    q_values, policy, metrics = td.q_learning(
        alpha=0.1,
        gamma=0.9,
        epsilon=0.3,
        episodes=episodes,
        decay_epsilon=True,
        decay_alpha=False
    )
    
    # Visualize results
    print("\nQ-Values:")
    td.visualize_q_values(q_values, title=f"{title_prefix} - Q-Values")
    
    print("\nLearned Policy:")
    td.visualize_policy(policy, title=f"{title_prefix} - Policy")
    
    print("\nLearning Curve:")
    td.visualize_learning_curve(metrics, title=f"{title_prefix} - Learning Progression")
    
    print("\nSuccess Rate:")
    td.visualize_success_rate(metrics, title=f"{title_prefix} - Success Rate")
    
    # Convert policy to dictionary format expected by the environment
    policy_dict = {}
    for state, action in policy.items():
        policy_dict[state] = action
    
    # Run a test episode with the learned policy
    total_reward, steps, states = environment.run_episode(policy=policy_dict)
    print(f"\nTest Episode Results:")
    print(f"Total reward: {total_reward}")
    print(f"Steps taken: {steps}")
    
    # Visualize the episode
    environment.render(agent_history=True)
    
    return q_values, policy, metrics

def test_sarsa(environment, episodes=500, title_prefix="SARSA"):
    """
    Test the SARSA algorithm.
    
    Args:
        environment: The GridWorldEnvironment
        episodes (int): Number of episodes to run
        title_prefix (str): Prefix for plot titles
    """
    print(f"Testing {title_prefix}...")
    
    # Create TD Learning instance
    td = TDLearning(environment)
    
    # Run SARSA with decaying epsilon
    q_values, policy, metrics = td.sarsa(
        alpha=0.1,
        gamma=0.9,
        epsilon=0.3,
        episodes=episodes,
        decay_epsilon=True,
        decay_alpha=False
    )
    
    # Visualize results
    print("\nQ-Values:")
    td.visualize_q_values(q_values, title=f"{title_prefix} - Q-Values")
    
    print("\nLearned Policy:")
    td.visualize_policy(policy, title=f"{title_prefix} - Policy")
    
    print("\nLearning Curve:")
    td.visualize_learning_curve(metrics, title=f"{title_prefix} - Learning Progression")
    
    print("\nSuccess Rate:")
    td.visualize_success_rate(metrics, title=f"{title_prefix} - Success Rate")
    
    # Convert policy to dictionary format expected by the environment
    policy_dict = {}
    for state, action in policy.items():
        policy_dict[state] = action
    
    # Run a test episode with the learned policy
    total_reward, steps, states = environment.run_episode(policy=policy_dict)
    print(f"\nTest Episode Results:")
    print(f"Total reward: {total_reward}")
    print(f"Steps taken: {steps}")
    
    # Visualize the episode
    environment.render(agent_history=True)
    
    return q_values, policy, metrics

def compare_algorithms(environment, episodes=500, title_prefix="Comparison"):
    """
    Compare Q-Learning and SARSA algorithms.
    
    Args:
        environment: The GridWorldEnvironment
        episodes (int): Number of episodes to run
        title_prefix (str): Prefix for plot titles
    """
    print(f"Comparing Q-Learning and SARSA in {title_prefix} Environment...")
    
    # Create TD Learning instance
    td = TDLearning(environment)
    
    # Run Q-Learning
    _, _, q_metrics = td.q_learning(
        alpha=0.1,
        gamma=0.9,
        epsilon=0.3,
        episodes=episodes,
        decay_epsilon=True
    )
    
    # Run SARSA
    _, _, sarsa_metrics = td.sarsa(
        alpha=0.1,
        gamma=0.9,
        epsilon=0.3,
        episodes=episodes,
        decay_epsilon=True
    )
    
    # Compare algorithms
    td.compare_algorithms(q_metrics, sarsa_metrics, title=f"{title_prefix} - Q-Learning vs SARSA")

def test_parameter_sensitivity(environment, episodes=300, parameter_name="epsilon"):
    """
    Test sensitivity to a specific parameter.
    
    Args:
        environment: The GridWorldEnvironment
        episodes (int): Number of episodes per run
        parameter_name (str): Parameter to test (epsilon, alpha, gamma)
    """
    print(f"Testing sensitivity to {parameter_name}...")
    
    # Create TD Learning instance
    td = TDLearning(environment)
    
    # Define parameter values to test
    if parameter_name == "epsilon":
        param_values = [0.01, 0.1, 0.3, 0.5, 0.8]
        title = "Exploration Rate Sensitivity"
    elif parameter_name == "alpha":
        param_values = [0.01, 0.05, 0.1, 0.3, 0.5]
        title = "Learning Rate Sensitivity"
    elif parameter_name == "gamma":
        param_values = [0.5, 0.7, 0.9, 0.95, 0.99]
        title = "Discount Factor Sensitivity"
    else:
        raise ValueError(f"Unknown parameter: {parameter_name}")
    
    # Run Q-Learning with different parameter values
    results = []
    
    for value in param_values:
        print(f"Testing {parameter_name}={value}...")
        
        # Set parameters
        kwargs = {
            "alpha": 0.1,
            "gamma": 0.9,
            "epsilon": 0.3,
            "episodes": episodes,
            "decay_epsilon": True
        }
        
        # Override the parameter we're testing
        kwargs[parameter_name] = value
        
        # Run Q-Learning
        _, _, metrics = td.q_learning(**kwargs)
        
        # Store results
        results.append((value, metrics))
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    # Plot episode rewards
    plt.subplot(2, 1, 1)
    for value, metrics in results:
        # Smooth rewards with moving average
        window_size = min(10, len(metrics['episode_rewards']))
        if window_size > 0:
            moving_avg = np.convolve(metrics['episode_rewards'], 
                                   np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(metrics['episode_rewards'])), 
                    moving_avg, label=f"{parameter_name}={value}")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(f"Reward vs Episode for Different {parameter_name} Values")
    plt.legend()
    plt.grid(True)
    
    # Plot final performance metrics
    plt.subplot(2, 1, 2)
    
    # Calculate average rewards in the last 10% of episodes
    final_rewards = []
    final_success_rates = []
    
    for value, metrics in results:
        last_index = int(0.9 * len(metrics['episode_rewards']))
        avg_reward = np.mean(metrics['episode_rewards'][last_index:])
        final_rewards.append(avg_reward)
        
        success = (np.array(metrics['episode_success'][last_index:]) == 1).astype(int)
        success_rate = np.mean(success)
        final_success_rates.append(success_rate)
    
    # Plot final metrics
    x = range(len(param_values))
    width = 0.35
    
    plt.bar(x, final_rewards, width, label="Avg Reward")
    plt.bar([i + width for i in x], final_success_rates, width, label="Success Rate")
    
    plt.xlabel(parameter_name)
    plt.ylabel("Performance Metric")
    plt.title(f"Final Performance vs {parameter_name}")
    plt.xticks([i + width/2 for i in x], param_values)
    plt.legend()
    plt.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

def test_stochastic_environment():
    """Test both algorithms in a stochastic environment."""
    print("Testing in a stochastic environment...\n")
    
    # Create a stochastic environment
    env = create_test_environment(is_stochastic=True)
    
    # Display environment
    print("Environment:")
    env.render()
    
    # Test both algorithms
    q_results = test_q_learning(env, episodes=500, title_prefix="Q-Learning (Stochastic)")
    print("\n" + "="*50 + "\n")
    sarsa_results = test_sarsa(env, episodes=500, title_prefix="SARSA (Stochastic)")
    print("\n" + "="*50 + "\n")
    
    # Compare algorithms
    compare_algorithms(env, episodes=500, title_prefix="Stochastic Environment")

def test_deterministic_environment():
    """Test both algorithms in a deterministic environment."""
    print("Testing in a deterministic environment...\n")
    
    # Create a deterministic environment
    env = create_test_environment(is_stochastic=False)
    
    # Display environment
    print("Environment:")
    env.render()
    
    # Test both algorithms
    q_results = test_q_learning(env, episodes=500, title_prefix="Q-Learning (Deterministic)")
    print("\n" + "="*50 + "\n")
    sarsa_results = test_sarsa(env, episodes=500, title_prefix="SARSA (Deterministic)")
    print("\n" + "="*50 + "\n")
    
    # Compare algorithms
    compare_algorithms(env, episodes=500, title_prefix="Deterministic Environment")

def test_parameter_sensitivity_all():
    """Test parameter sensitivity for all key parameters."""
    # Create a deterministic environment for consistent results
    env = create_test_environment(is_stochastic=False)
    
    # Test sensitivity to exploration rate
    test_parameter_sensitivity(env, parameter_name="epsilon")
    
    # Test sensitivity to learning rate
    test_parameter_sensitivity(env, parameter_name="alpha")
    
    # Test sensitivity to discount factor
    test_parameter_sensitivity(env, parameter_name="gamma")

def main():
    """Run all tests."""
    # Test in deterministic environment
    test_deterministic_environment()
    
    print("\n" + "="*70 + "\n")
    
    # Test in stochastic environment
    test_stochastic_environment()
    
    print("\n" + "="*70 + "\n")
    
    # Test parameter sensitivity
    test_parameter_sensitivity_all()

if __name__ == "__main__":
    main()