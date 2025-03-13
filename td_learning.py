import numpy as np
import matplotlib.pyplot as plt
import time
from GridWorldEnvironment import GridWorldEnvironment

class TDLearning:
    """
    Temporal Difference Learning methods for reinforcement learning.
    
    Implements Q-Learning and SARSA algorithms for learning optimal policies
    through experience without requiring a model of the environment.
    """
    
    def __init__(self, environment):
        """
        Initialize TDLearning with an environment.
        
        Args:
            environment: A GridWorldEnvironment instance
        """
        self.env = environment
    
    def epsilon_greedy(self, q_values, state, epsilon):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            q_values (dict): Q-values dictionary mapping states to {action: value} dicts
            state (tuple): Current state (row, col)
            epsilon (float): Probability of choosing a random action
            
        Returns:
            int: Selected action
        """
        # Get valid actions from the environment
        valid_actions = self.env.get_valid_actions(state)
        
        # With probability epsilon, choose a random action
        if np.random.random() < epsilon:
            return np.random.choice(valid_actions)
        
        # Otherwise, choose the action with the highest Q-value
        # If the state has not been seen before, initialize it in the Q-table
        if state not in q_values:
            q_values[state] = {action: 0.0 for action in valid_actions}
        
        # If all Q-values are equal (e.g., all 0), choose randomly
        if len(set(q_values[state].values())) == 1:
            return np.random.choice(valid_actions)
        
        # Choose the best action among valid actions
        return max(valid_actions, key=lambda a: q_values[state].get(a, 0.0))
    
    def q_learning(self, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=500, decay_epsilon=True, decay_alpha=False):
        """
        Implement Q-Learning algorithm.
        
        Args:
            alpha (float): Learning rate
            gamma (float): Discount factor
            epsilon (float): Initial exploration rate
            episodes (int): Number of episodes to run
            decay_epsilon (bool): Whether to decay epsilon over episodes
            decay_alpha (bool): Whether to decay alpha over episodes
            
        Returns:
            tuple: (q_table, policy, metrics)
                q_table: Dictionary mapping states to action-value dictionaries
                policy: Dictionary mapping states to best actions
                metrics: Dictionary containing performance metrics
        """
        # Initialize Q-values
        q_values = {}
        
        # Initialize metrics
        metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_success': [],  # 1: goal, -1: trap, 0: max steps
            'epsilon_values': [],
            'alpha_values': [],
            'avg_q_values': [],
            'time': 0
        }
        
        # Start timer
        start_time = time.time()
        
        # Run episodes
        for episode in range(episodes):
            # Decay epsilon and alpha if needed
            if decay_epsilon:
                # Start with more exploration, gradually reduce
                current_epsilon = max(0.01, epsilon * (1.0 - episode / episodes))
            else:
                current_epsilon = epsilon
                
            if decay_alpha:
                # Similar decay for learning rate
                current_alpha = max(0.01, alpha * (1.0 - episode / episodes))
            else:
                current_alpha = alpha
            
            # Track these values
            metrics['epsilon_values'].append(current_epsilon)
            metrics['alpha_values'].append(current_alpha)
            
            # Reset environment
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            # Initialize episode stats
            q_values_sum = 0
            q_values_count = 0
            
            # Run episode
            done = False
            while not done:
                # Select action using epsilon-greedy
                action = self.epsilon_greedy(q_values, state, current_epsilon)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Ensure the state is in the Q-table
                if state not in q_values:
                    q_values[state] = {a: 0.0 for a in self.env.get_valid_actions(state)}
                    
                # Ensure the next state is in the Q-table if not terminal
                if not done and next_state not in q_values:
                    q_values[next_state] = {a: 0.0 for a in self.env.get_valid_actions(next_state)}
                
                # Q-Learning update:
                # Q(s,a) = Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
                
                # Calculate the target (future value)
                if done:
                    # Terminal state, no future rewards
                    target = reward
                else:
                    # Non-terminal state, consider future rewards
                    next_max_q = max(q_values[next_state].values())
                    target = reward + gamma * next_max_q
                
                # Calculate the TD error
                td_error = target - q_values[state].get(action, 0.0)
                
                # Update the Q-value
                q_values[state][action] = q_values[state].get(action, 0.0) + current_alpha * td_error
                
                # Track average Q-values for this episode
                q_values_sum += q_values[state][action]
                q_values_count += 1
                
                # Move to next state
                state = next_state
                total_reward += reward
                steps += 1
            
            # Track episode metrics
            metrics['episode_rewards'].append(total_reward)
            metrics['episode_lengths'].append(steps)
            
            # Determine if episode was successful
            success_code = 0  # Default: reached max steps
            if info['is_goal']:
                success_code = 1  # Reached goal
            elif info['is_trap']:
                success_code = -1  # Fell into trap
                
            metrics['episode_success'].append(success_code)
            
            # Track average Q-value for this episode
            if q_values_count > 0:
                metrics['avg_q_values'].append(q_values_sum / q_values_count)
            else:
                metrics['avg_q_values'].append(0)
        
        # Record elapsed time
        metrics['time'] = time.time() - start_time
        
        # Extract policy from Q-values
        policy = {}
        for state, actions in q_values.items():
            # Only include non-terminal states with valid actions
            if actions:  # If the dictionary is not empty
                policy[state] = max(actions.items(), key=lambda x: x[1])[0]
        
        return q_values, policy, metrics
    
    def sarsa(self, alpha=0.1, gamma=0.9, epsilon=0.1, episodes=500, decay_epsilon=True, decay_alpha=False):
        """
        Implement SARSA algorithm.
        
        Args:
            alpha (float): Learning rate
            gamma (float): Discount factor
            epsilon (float): Initial exploration rate
            episodes (int): Number of episodes to run
            decay_epsilon (bool): Whether to decay epsilon over episodes
            decay_alpha (bool): Whether to decay alpha over episodes
            
        Returns:
            tuple: (q_table, policy, metrics)
                q_table: Dictionary mapping states to action-value dictionaries
                policy: Dictionary mapping states to best actions
                metrics: Dictionary containing performance metrics
        """
        # Initialize Q-values
        q_values = {}
        
        # Initialize metrics
        metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_success': [],  # 1: goal, -1: trap, 0: max steps
            'epsilon_values': [],
            'alpha_values': [],
            'avg_q_values': [],
            'time': 0
        }
        
        # Start timer
        start_time = time.time()
        
        # Run episodes
        for episode in range(episodes):
            # Decay epsilon and alpha if needed
            if decay_epsilon:
                # Start with more exploration, gradually reduce
                current_epsilon = max(0.01, epsilon * (1.0 - episode / episodes))
            else:
                current_epsilon = epsilon
                
            if decay_alpha:
                # Similar decay for learning rate
                current_alpha = max(0.01, alpha * (1.0 - episode / episodes))
            else:
                current_alpha = alpha
            
            # Track these values
            metrics['epsilon_values'].append(current_epsilon)
            metrics['alpha_values'].append(current_alpha)
            
            # Reset environment
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            # Initialize episode stats
            q_values_sum = 0
            q_values_count = 0
            
            # Ensure the state is in the Q-table
            if state not in q_values:
                q_values[state] = {a: 0.0 for a in self.env.get_valid_actions(state)}
            
            # Choose initial action using epsilon-greedy
            action = self.epsilon_greedy(q_values, state, current_epsilon)
            
            # Run episode
            done = False
            while not done:
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Ensure the next state is in the Q-table if not terminal
                if not done and next_state not in q_values:
                    q_values[next_state] = {a: 0.0 for a in self.env.get_valid_actions(next_state)}
                
                # Choose next action using epsilon-greedy
                if not done:
                    next_action = self.epsilon_greedy(q_values, next_state, current_epsilon)
                else:
                    next_action = None  # No next action for terminal states
                
                # SARSA update:
                # Q(s,a) = Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
                
                # Calculate the target (future value)
                if done:
                    # Terminal state, no future rewards
                    target = reward
                else:
                    # Non-terminal state, consider future rewards from the chosen action
                    target = reward + gamma * q_values[next_state].get(next_action, 0.0)
                
                # Calculate the TD error
                td_error = target - q_values[state].get(action, 0.0)
                
                # Update the Q-value
                q_values[state][action] = q_values[state].get(action, 0.0) + current_alpha * td_error
                
                # Track average Q-values for this episode
                q_values_sum += q_values[state][action]
                q_values_count += 1
                
                # Move to next state and action
                state = next_state
                action = next_action
                total_reward += reward
                steps += 1
                
                # If terminal state, break the loop
                if done:
                    break
            
            # Track episode metrics
            metrics['episode_rewards'].append(total_reward)
            metrics['episode_lengths'].append(steps)
            
            # Determine if episode was successful
            success_code = 0  # Default: reached max steps
            if info['is_goal']:
                success_code = 1  # Reached goal
            elif info['is_trap']:
                success_code = -1  # Fell into trap
                
            metrics['episode_success'].append(success_code)
            
            # Track average Q-value for this episode
            if q_values_count > 0:
                metrics['avg_q_values'].append(q_values_sum / q_values_count)
            else:
                metrics['avg_q_values'].append(0)
        
        # Record elapsed time
        metrics['time'] = time.time() - start_time
        
        # Extract policy from Q-values
        policy = {}
        for state, actions in q_values.items():
            # Only include non-terminal states with valid actions
            if actions:  # If the dictionary is not empty
                policy[state] = max(actions.items(), key=lambda x: x[1])[0]
        
        return q_values, policy, metrics
    
    def visualize_policy(self, policy, title="Learned Policy"):
        """
        Visualize the learned policy as arrows on a grid.
        
        Args:
            policy (dict): Policy mapping states to actions
            title (str): Plot title
        """
        # Create a 2D grid for the policy
        grid_policy = np.ones((self.env.height, self.env.width), dtype=int) * -1
        
        # Fill in the policy
        for state, action in policy.items():
            r, c = state
            grid_policy[r, c] = action
        
        # Visualize the policy using the environment's render method
        self.env.render(policy=grid_policy)
        plt.title(title)
        plt.show()
    
    def visualize_q_values(self, q_values, title="Q-Values Heatmap"):
        """
        Visualize the Q-values as a heatmap.
        
        Args:
            q_values (dict): Q-values dictionary mapping states to {action: value} dicts
            title (str): Plot title
        """
        # Create a 2D grid for the state values (using max Q-value for each state)
        grid_values = np.zeros((self.env.height, self.env.width))
        
        # Fill in the values
        for r in range(self.env.height):
            for c in range(self.env.width):
                state = (r, c)
                if state in q_values and q_values[state]:
                    grid_values[r, c] = max(q_values[state].values())
                else:
                    grid_values[r, c] = np.nan
        
        # Use the environment's render method to show values
        self.env.render(show_values=True, values=grid_values)
        plt.title(title)
        plt.show()
    
    def visualize_learning_curve(self, metrics, title="Learning Curve"):
        """
        Visualize the learning curve (rewards over episodes).
        
        Args:
            metrics (dict): Metrics from learning algorithm
            title (str): Plot title
        """
        plt.figure(figsize=(12, 8))
        
        # Plot episode rewards
        plt.subplot(2, 2, 1)
        plt.plot(metrics['episode_rewards'], label="Episode Rewards")
        # Add moving average for smoothing
        window_size = min(10, len(metrics['episode_rewards']))
        if window_size > 0:
            moving_avg = np.convolve(metrics['episode_rewards'], 
                                   np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(metrics['episode_rewards'])), 
                    moving_avg, 'r', label=f"Moving Avg ({window_size})")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Episode Rewards")
        plt.legend()
        plt.grid(True)
        
        # Plot episode lengths
        plt.subplot(2, 2, 2)
        plt.plot(metrics['episode_lengths'], label="Episode Length")
        # Add moving average for smoothing
        if window_size > 0:
            moving_avg = np.convolve(metrics['episode_lengths'], 
                                   np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(metrics['episode_lengths'])), 
                    moving_avg, 'r', label=f"Moving Avg ({window_size})")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.title("Episode Lengths")
        plt.legend()
        plt.grid(True)
        
        # Plot epsilon decay
        plt.subplot(2, 2, 3)
        plt.plot(metrics['epsilon_values'], label="Epsilon")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon Value")
        plt.title("Exploration Rate")
        plt.grid(True)
        
        # Plot average Q-values
        plt.subplot(2, 2, 4)
        plt.plot(metrics['avg_q_values'], label="Avg Q-Value")
        plt.xlabel("Episode")
        plt.ylabel("Average Q-Value")
        plt.title("Q-Value Progression")
        plt.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()
    
    def visualize_success_rate(self, metrics, window_size=10, title="Success Rate"):
        """
        Visualize the success rate over episodes.
        
        Args:
            metrics (dict): Metrics from learning algorithm
            window_size (int): Window size for moving average
            title (str): Plot title
        """
        plt.figure(figsize=(10, 6))
        
        # Convert success codes to success rate
        # 1: reached goal, -1: fell into trap, 0: reached max steps
        success_values = np.array(metrics['episode_success'])
        goal_reached = (success_values == 1).astype(int)
        
        # Calculate moving average of success rate
        if len(goal_reached) >= window_size:
            moving_avg = np.convolve(goal_reached, 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            plt.plot(range(window_size-1, len(goal_reached)), 
                    moving_avg, 'g', label=f"Success Rate (window={window_size})")
        
        plt.xlabel("Episode")
        plt.ylabel("Success Rate")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def compare_algorithms(self, q_metrics, sarsa_metrics, title="Q-Learning vs SARSA"):
        """
        Compare the performance of Q-Learning and SARSA.
        
        Args:
            q_metrics (dict): Metrics from Q-Learning
            sarsa_metrics (dict): Metrics from SARSA
            title (str): Plot title
        """
        plt.figure(figsize=(15, 10))
        
        # Plot episode rewards
        plt.subplot(2, 2, 1)
        plt.plot(q_metrics['episode_rewards'], label="Q-Learning")
        plt.plot(sarsa_metrics['episode_rewards'], label="SARSA")
        # Add moving averages
        window_size = min(10, len(q_metrics['episode_rewards']))
        if window_size > 0:
            q_moving_avg = np.convolve(q_metrics['episode_rewards'], 
                                     np.ones(window_size)/window_size, mode='valid')
            sarsa_moving_avg = np.convolve(sarsa_metrics['episode_rewards'], 
                                         np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(q_metrics['episode_rewards'])), 
                    q_moving_avg, 'r--', label=f"Q-Learning Avg ({window_size})")
            plt.plot(range(window_size-1, len(sarsa_metrics['episode_rewards'])), 
                    sarsa_moving_avg, 'g--', label=f"SARSA Avg ({window_size})")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Episode Rewards")
        plt.legend()
        plt.grid(True)
        
        # Plot episode lengths
        plt.subplot(2, 2, 2)
        plt.plot(q_metrics['episode_lengths'], label="Q-Learning")
        plt.plot(sarsa_metrics['episode_lengths'], label="SARSA")
        # Add moving averages
        if window_size > 0:
            q_moving_avg = np.convolve(q_metrics['episode_lengths'], 
                                     np.ones(window_size)/window_size, mode='valid')
            sarsa_moving_avg = np.convolve(sarsa_metrics['episode_lengths'], 
                                         np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(q_metrics['episode_lengths'])), 
                    q_moving_avg, 'r--', label=f"Q-Learning Avg ({window_size})")
            plt.plot(range(window_size-1, len(sarsa_metrics['episode_lengths'])), 
                    sarsa_moving_avg, 'g--', label=f"SARSA Avg ({window_size})")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.title("Episode Lengths")
        plt.legend()
        plt.grid(True)
        
        # Plot average Q-values
        plt.subplot(2, 2, 3)
        plt.plot(q_metrics['avg_q_values'], label="Q-Learning")
        plt.plot(sarsa_metrics['avg_q_values'], label="SARSA")
        plt.xlabel("Episode")
        plt.ylabel("Average Q-Value")
        plt.title("Q-Value Progression")
        plt.legend()
        plt.grid(True)
        
        # Plot success rate
        plt.subplot(2, 2, 4)
        # Convert success codes to success rate (1: reached goal)
        q_success = (np.array(q_metrics['episode_success']) == 1).astype(int)
        sarsa_success = (np.array(sarsa_metrics['episode_success']) == 1).astype(int)
        
        # Calculate moving average of success rate
        if len(q_success) >= window_size:
            q_success_avg = np.convolve(q_success, 
                                      np.ones(window_size)/window_size, 
                                      mode='valid')
            sarsa_success_avg = np.convolve(sarsa_success, 
                                          np.ones(window_size)/window_size, 
                                          mode='valid')
            plt.plot(range(window_size-1, len(q_success)), 
                    q_success_avg, 'r', label=f"Q-Learning Success")
            plt.plot(range(window_size-1, len(sarsa_success)), 
                    sarsa_success_avg, 'g', label=f"SARSA Success")
        
        plt.xlabel("Episode")
        plt.ylabel("Success Rate")
        plt.title("Goal Reached Rate")
        plt.legend()
        plt.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()
        
        # Print comparison summary
        print(f"Performance Comparison: Q-Learning vs SARSA")
        print(f"Time taken - Q-Learning: {q_metrics['time']:.3f}s, SARSA: {sarsa_metrics['time']:.3f}s")
        
        # Compare final success rate (last 10% of episodes)
        last_index = int(0.9 * len(q_success))
        q_final_success = np.mean(q_success[last_index:])
        sarsa_final_success = np.mean(sarsa_success[last_index:])
        print(f"Final success rate - Q-Learning: {q_final_success:.2f}, SARSA: {sarsa_final_success:.2f}")
        
        # Compare final average rewards
        q_final_rewards = np.mean(q_metrics['episode_rewards'][last_index:])
        sarsa_final_rewards = np.mean(sarsa_metrics['episode_rewards'][last_index:])
        print(f"Final average reward - Q-Learning: {q_final_rewards:.2f}, SARSA: {sarsa_final_rewards:.2f}")
        
        # Compare final episode lengths
        q_final_lengths = np.mean(q_metrics['episode_lengths'][last_index:])
        sarsa_final_lengths = np.mean(sarsa_metrics['episode_lengths'][last_index:])
        print(f"Final average episode length - Q-Learning: {q_final_lengths:.2f}, SARSA: {sarsa_final_lengths:.2f}")