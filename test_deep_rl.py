import numpy as np
import matplotlib.pyplot as plt
from GridWorldEnvironment import GridWorldEnvironment
from deep_rl import DQN, dqn

def create_test_environment(is_stochastic=False):
    """
    Create a test environment for the DQN algorithm.
    
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

def test_dqn(environment, episodes=300, title_prefix="DQN"):
    """
    Test the DQN algorithm.
    
    Args:
        environment: The GridWorldEnvironment
        episodes (int): Number of episodes to run
        title_prefix (str): Prefix for plot titles
    """
    print(f"Testing {title_prefix}...")
    
    # Run DQN
    dqn_agent, policy, metrics = dqn(
        environment=environment,
        hidden_layers=[64, 64],
        gamma=0.99,
        epsilon=1.0,
        episodes=episodes
    )
    
    # Visualize results
    print("\nQ-Values:")
    dqn_agent.visualize_q_values(title=f"{title_prefix} - Q-Values")
    
    print("\nLearned Policy:")
    dqn_agent.visualize_policy(title=f"{title_prefix} - Policy")
    
    print("\nLearning Curve:")
    dqn_agent.visualize_learning_curve(title=f"{title_prefix} - Learning Progression")
    
    print("\nSuccess Rate:")
    dqn_agent.visualize_success_rate(title=f"{title_prefix} - Success Rate")
    
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
    
    return dqn_agent, policy, metrics

def compare_network_architectures(environment, episodes=200):
    """
    Compare different neural network architectures.
    
    Args:
        environment: The GridWorldEnvironment
        episodes (int): Number of episodes for each run
    """
    print("Comparing different neural network architectures...")
    
    # Define architectures to test
    architectures = [
        [32],
        [64],
        [32, 32],
        [64, 64],
        [128, 64, 32]
    ]
    
    # Track metrics for each architecture
    all_metrics = []
    
    for hidden_layers in architectures:
        print(f"\nTesting architecture: {hidden_layers}")
        
        # Calculate state and action space sizes
        state_size = environment.height * environment.width
        action_size = len(environment.get_all_actions())
        
        # Create DQN agent
        dqn_agent = DQN(
            environment=environment,
            state_size=state_size,
            action_size=action_size,
            hidden_layers=hidden_layers,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=64,
            target_update_freq=10
        )
        
        # Train the agent
        metrics = dqn_agent.train(episodes=episodes, render_freq=None)
        
        # Store metrics
        all_metrics.append((hidden_layers, metrics))
    
    # Visualize comparison
    plt.figure(figsize=(15, 10))
    
    # Plot episode rewards
    plt.subplot(2, 2, 1)
    for hidden_layers, metrics in all_metrics:
        label = str(hidden_layers)
        window_size = min(10, len(metrics['episode_rewards']))
        if window_size > 0:
            moving_avg = np.convolve(metrics['episode_rewards'], 
                                   np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(metrics['episode_rewards'])), 
                    moving_avg, label=f"Layers: {label}")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Reward vs Episode for Different Architectures")
    plt.legend()
    plt.grid(True)
    
    # Plot success rates
    plt.subplot(2, 2, 2)
    window_size = 10
    for hidden_layers, metrics in all_metrics:
        label = str(hidden_layers)
        success_values = np.array(metrics['episode_success'])
        goal_reached = (success_values == 1).astype(int)
        
        if len(goal_reached) >= window_size:
            moving_avg = np.convolve(goal_reached, 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            plt.plot(range(window_size-1, len(goal_reached)), 
                    moving_avg, label=f"Layers: {label}")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.title("Success Rate for Different Architectures")
    plt.legend()
    plt.grid(True)
    
    # Plot training loss
    plt.subplot(2, 2, 3)
    for hidden_layers, metrics in all_metrics:
        label = str(hidden_layers)
        plt.plot(metrics['losses'], label=f"Layers: {label}")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training Loss for Different Architectures")
    plt.legend()
    plt.grid(True)
    
    # Plot final metrics as bar chart
    plt.subplot(2, 2, 4)
    
    # Calculate average rewards and success rates in last 10% of episodes
    final_rewards = []
    final_success_rates = []
    labels = []
    
    for hidden_layers, metrics in all_metrics:
        labels.append(str(hidden_layers))
        
        last_index = int(0.9 * len(metrics['episode_rewards']))
        avg_reward = np.mean(metrics['episode_rewards'][last_index:])
        final_rewards.append(avg_reward)
        
        success = (np.array(metrics['episode_success'][last_index:]) == 1).astype(int)
        success_rate = np.mean(success)
        final_success_rates.append(success_rate)
    
    # Plot final metrics
    x = range(len(labels))
    width = 0.35
    
    plt.bar(x, final_rewards, width, label="Avg Reward")
    plt.bar([i + width for i in x], final_success_rates, width, label="Success Rate")
    
    plt.xlabel("Architecture")
    plt.ylabel("Performance Metric")
    plt.title("Final Performance vs Architecture")
    plt.xticks([i + width/2 for i in x], labels)
    plt.legend()
    plt.grid(True)
    
    plt.suptitle("Neural Network Architecture Comparison")
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

def compare_buffer_sizes(environment, episodes=200):
    """
    Compare different replay buffer sizes.
    
    Args:
        environment: The GridWorldEnvironment
        episodes (int): Number of episodes for each run
    """
    print("Comparing different replay buffer sizes...")
    
    # Define buffer sizes to test
    buffer_sizes = [1000, 5000, 10000, 20000]
    
    # Track metrics for each buffer size
    all_metrics = []
    
    for buffer_size in buffer_sizes:
        print(f"\nTesting buffer size: {buffer_size}")
        
        # Calculate state and action space sizes
        state_size = environment.height * environment.width
        action_size = len(environment.get_all_actions())
        
        # Create DQN agent
        dqn_agent = DQN(
            environment=environment,
            state_size=state_size,
            action_size=action_size,
            hidden_layers=[64, 64],
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            buffer_size=buffer_size,
            batch_size=64,
            target_update_freq=10
        )
        
        # Train the agent
        metrics = dqn_agent.train(episodes=episodes, render_freq=None)
        
        # Store metrics
        all_metrics.append((buffer_size, metrics))
    
    # Visualize comparison
    plt.figure(figsize=(15, 10))
    
    # Plot episode rewards
    plt.subplot(2, 2, 1)
    for buffer_size, metrics in all_metrics:
        window_size = min(10, len(metrics['episode_rewards']))
        if window_size > 0:
            moving_avg = np.convolve(metrics['episode_rewards'], 
                                   np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(metrics['episode_rewards'])), 
                    moving_avg, label=f"Buffer: {buffer_size}")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Reward vs Episode for Different Buffer Sizes")
    plt.legend()
    plt.grid(True)
    
    # Plot success rates
    plt.subplot(2, 2, 2)
    window_size = 10
    for buffer_size, metrics in all_metrics:
        success_values = np.array(metrics['episode_success'])
        goal_reached = (success_values == 1).astype(int)
        
        if len(goal_reached) >= window_size:
            moving_avg = np.convolve(goal_reached, 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            plt.plot(range(window_size-1, len(goal_reached)), 
                    moving_avg, label=f"Buffer: {buffer_size}")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.title("Success Rate for Different Buffer Sizes")
    plt.legend()
    plt.grid(True)
    
    # Plot training loss
    plt.subplot(2, 2, 3)
    for buffer_size, metrics in all_metrics:
        plt.plot(metrics['losses'], label=f"Buffer: {buffer_size}")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training Loss for Different Buffer Sizes")
    plt.legend()
    plt.grid(True)
    
    # Plot final metrics as bar chart
    plt.subplot(2, 2, 4)
    
    # Calculate average rewards and success rates in last 10% of episodes
    final_rewards = []
    final_success_rates = []
    labels = []
    
    for buffer_size, metrics in all_metrics:
        labels.append(str(buffer_size))
        
        last_index = int(0.9 * len(metrics['episode_rewards']))
        avg_reward = np.mean(metrics['episode_rewards'][last_index:])
        final_rewards.append(avg_reward)
        
        success = (np.array(metrics['episode_success'][last_index:]) == 1).astype(int)
        success_rate = np.mean(success)
        final_success_rates.append(success_rate)
    
    # Plot final metrics
    x = range(len(labels))
    width = 0.35
    
    plt.bar(x, final_rewards, width, label="Avg Reward")
    plt.bar([i + width for i in x], final_success_rates, width, label="Success Rate")
    
    plt.xlabel("Buffer Size")
    plt.ylabel("Performance Metric")
    plt.title("Final Performance vs Buffer Size")
    plt.xticks([i + width/2 for i in x], labels)
    plt.legend()
    plt.grid(True)
    
    plt.suptitle("Replay Buffer Size Comparison")
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

def compare_update_frequencies(environment, episodes=200):
    """
    Compare different target network update frequencies.
    
    Args:
        environment: The GridWorldEnvironment
        episodes (int): Number of episodes for each run
    """
    print("Comparing different target network update frequencies...")
    
    # Define update frequencies to test
    update_freqs = [1, 5, 10, 20]
    
    # Track metrics for each update frequency
    all_metrics = []
    
    for update_freq in update_freqs:
        print(f"\nTesting update frequency: {update_freq}")
        
        # Calculate state and action space sizes
        state_size = environment.height * environment.width
        action_size = len(environment.get_all_actions())
        
        # Create DQN agent
        dqn_agent = DQN(
            environment=environment,
            state_size=state_size,
            action_size=action_size,
            hidden_layers=[64, 64],
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            buffer_size=10000,
            batch_size=64,
            target_update_freq=update_freq
        )
        
        # Train the agent
        metrics = dqn_agent.train(episodes=episodes, render_freq=None)
        
        # Store metrics
        all_metrics.append((update_freq, metrics))
    
    # Visualize comparison
    plt.figure(figsize=(15, 10))
    
    # Plot episode rewards
    plt.subplot(2, 2, 1)
    for update_freq, metrics in all_metrics:
        window_size = min(10, len(metrics['episode_rewards']))
        if window_size > 0:
            moving_avg = np.convolve(metrics['episode_rewards'], 
                                   np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(metrics['episode_rewards'])), 
                    moving_avg, label=f"Update Freq: {update_freq}")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Reward vs Episode for Different Update Frequencies")
    plt.legend()
    plt.grid(True)
    
    # Plot success rates
    plt.subplot(2, 2, 2)
    window_size = 10
    for update_freq, metrics in all_metrics:
        success_values = np.array(metrics['episode_success'])
        goal_reached = (success_values == 1).astype(int)
        
        if len(goal_reached) >= window_size:
            moving_avg = np.convolve(goal_reached, 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            plt.plot(range(window_size-1, len(goal_reached)), 
                    moving_avg, label=f"Update Freq: {update_freq}")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    plt.title("Success Rate for Different Update Frequencies")
    plt.legend()
    plt.grid(True)
    
    # Plot training loss
    plt.subplot(2, 2, 3)
    for update_freq, metrics in all_metrics:
        plt.plot(metrics['losses'], label=f"Update Freq: {update_freq}")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Training Loss for Different Update Frequencies")
    plt.legend()
    plt.grid(True)
    
    # Plot final metrics as bar chart
    plt.subplot(2, 2, 4)
    
    # Calculate average rewards and success rates in last 10% of episodes
    final_rewards = []
    final_success_rates = []
    labels = []
    
    for update_freq, metrics in all_metrics:
        labels.append(str(update_freq))
        
        last_index = int(0.9 * len(metrics['episode_rewards']))
        avg_reward = np.mean(metrics['episode_rewards'][last_index:])
        final_rewards.append(avg_reward)
        
        success = (np.array(metrics['episode_success'][last_index:]) == 1).astype(int)
        success_rate = np.mean(success)
        final_success_rates.append(success_rate)
    
    # Plot final metrics
    x = range(len(labels))
    width = 0.35
    
    plt.bar(x, final_rewards, width, label="Avg Reward")
    plt.bar([i + width for i in x], final_success_rates, width, label="Success Rate")
    
    plt.xlabel("Update Frequency")
    plt.ylabel("Performance Metric")
    plt.title("Final Performance vs Update Frequency")
    plt.xticks([i + width/2 for i in x], labels)
    plt.legend()
    plt.grid(True)
    
    plt.suptitle("Target Network Update Frequency Comparison")
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()

def create_dynamic_environment():
    """Create a dynamic environment where elements change based on agent actions."""
    # Create a 6x6 grid
    grid = np.zeros((6, 6))
    
    # Add walls
    walls = [(1, 1), (1, 4), (4, 1), (4, 4)]
    for r, c in walls:
        grid[r, c] = -100
    
    start_state = (0, 0)
    goal_states = [(5, 5)]
    trap_states = [(2, 2), (3, 3)]
    
    class DynamicGridWorld(GridWorldEnvironment):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dynamic_traps = [(2, 3), (3, 2)]
            self.trap_activation_count = 0
            
        def step(self, action):
            next_state, reward, done, info = super().step(action)
            
            # After every 5 steps, move the trap
            if self.steps % 5 == 0 and self.trap_activation_count < len(self.dynamic_traps):
                new_trap = self.dynamic_traps[self.trap_activation_count]
                if new_trap not in self.trap_states:
                    self.trap_states.append(new_trap)
                    self.grid[new_trap] = -self.trap_penalty
                    self.trap_activation_count += 1
            
            return next_state, reward, done, info
    
    # Create dynamic environment
    env = DynamicGridWorld(
        grid=grid,
        start_state=start_state,
        goal_states=goal_states,
        trap_states=trap_states,
        is_stochastic=True,
        transition_prob=0.8,
        step_cost=-0.1,
        goal_reward=1.0,
        trap_penalty=1.0
    )
    
    return env

def create_multi_goal_environment():
    """Create an environment with multiple goals of different rewards."""
    # Create a 7x7 grid
    grid = np.zeros((7, 7))
    
    # Add walls to create a more complex layout
    walls = [(1, 1), (1, 5), (2, 3), (3, 1), (3, 5), (5, 2), (5, 4)]
    for r, c in walls:
        grid[r, c] = -100
    
    start_state = (0, 0)
    
    # Multiple goals with different rewards
    goal_states = [(6, 6), (0, 6), (6, 0)]
    trap_states = [(3, 3)]
    
    # Custom rewards for different goals
    custom_rewards = {}
    
    class MultiGoalGridWorld(GridWorldEnvironment):
        def _calculate_reward(self, state, action, next_state):
            # Override reward calculation to provide different rewards for different goals
            if next_state == (6, 6):
                return 1.0  # High reward for the far corner
            elif next_state == (0, 6):
                return 0.7  # Medium reward for the top-right corner
            elif next_state == (6, 0):
                return 0.5  # Smaller reward for the bottom-left corner
            
            # For other states, use the default calculation
            return super()._calculate_reward(state, action, next_state)
    
    # Create multi-goal environment
    env = MultiGoalGridWorld(
        grid=grid,
        start_state=start_state,
        goal_states=goal_states,
        trap_states=trap_states,
        is_stochastic=True,
        transition_prob=0.8,
        step_cost=-0.05,
        goal_reward=1.0,  # This will be overridden by the custom reward calculation
        trap_penalty=1.0,
        custom_rewards=custom_rewards
    )
    
    return env

def main():
    """Run all tests."""
    # Test in deterministic environment
    deterministic_env = create_test_environment(is_stochastic=False)
    test_dqn(deterministic_env, episodes=300, title_prefix="DQN (Deterministic)")
    
    print("\n" + "="*70 + "\n")
    
    # Test in stochastic environment
    stochastic_env = create_test_environment(is_stochastic=True)
    test_dqn(stochastic_env, episodes=300, title_prefix="DQN (Stochastic)")
    
    print("\n" + "="*70 + "\n")
    
    # Compare neural network architectures
    compare_network_architectures(deterministic_env, episodes=200)
    
    print("\n" + "="*70 + "\n")
    
    # Compare replay buffer sizes
    compare_buffer_sizes(deterministic_env, episodes=200)
    
    print("\n" + "="*70 + "\n")
    
    # Compare target network update frequencies
    compare_update_frequencies(deterministic_env, episodes=200)
    
    print("\n" + "="*70 + "\n")
    
    # Test in dynamic environment
    dynamic_env = create_dynamic_environment()
    test_dqn(dynamic_env, episodes=300, title_prefix="DQN (Dynamic)")
    
    print("\n" + "="*70 + "\n")
    
    # Test in multi-goal environment
    multi_goal_env = create_multi_goal_environment()
    test_dqn(multi_goal_env, episodes=400, title_prefix="DQN (Multi-Goal)")

if __name__ == "__main__":
    main()