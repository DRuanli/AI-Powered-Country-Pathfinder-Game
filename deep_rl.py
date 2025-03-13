import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import time
import random
from collections import deque
from GridWorldEnvironment import GridWorldEnvironment

class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    """
    
    def __init__(self, capacity):
        """
        Initialize buffer with fixed capacity.
        
        Args:
            capacity (int): Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones)
        """
        # Ensure we have enough samples
        batch_size = min(batch_size, len(self.buffer))
        
        # Sample random batch
        batch = random.sample(self.buffer, batch_size)
        
        # Unzip the batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)


class DQN:
    """
    Deep Q-Network implementation for reinforcement learning.
    
    Uses neural networks to approximate the Q-function, allowing for better
    generalization in large state spaces.
    """
    
    def __init__(self, environment, state_size, action_size, hidden_layers=[64, 64], 
                 learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, 
                 epsilon_decay=0.995, buffer_size=10000, batch_size=64, target_update_freq=5):
        """
        Initialize DQN agent.
        
        Args:
            environment: The environment to interact with
            state_size (int): Size of the state representation
            action_size (int): Number of possible actions
            hidden_layers (list): Sizes of hidden layers in the neural network
            learning_rate (float): Learning rate for the optimizer
            gamma (float): Discount factor
            epsilon (float): Initial exploration rate
            epsilon_min (float): Minimum exploration rate
            epsilon_decay (float): Decay rate for exploration
            buffer_size (int): Size of the replay buffer
            batch_size (int): Batch size for training
            target_update_freq (int): Frequency of target network updates
        """
        self.env = environment
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Build neural networks
        self.main_network = self._build_network()
        self.target_network = self._build_network()
        
        # Initialize target network with main network's weights
        self.update_target_network()
        
        # Metrics for tracking performance
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_success': [],
            'epsilon_values': [],
            'losses': [],
            'avg_q_values': [],
            'time': 0
        }
    
    def _build_network(self):
        """
        Build a neural network for Q-function approximation.
        
        Returns:
            tf.keras.Model: Neural network model
        """
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.hidden_layers[0], input_dim=self.state_size, activation='relu'))
        
        # Hidden layers
        for layer_size in self.hidden_layers[1:]:
            model.add(Dense(layer_size, activation='relu'))
        
        # Output layer (one output per action)
        model.add(Dense(self.action_size, activation='linear'))
        
        # Compile model
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def update_target_network(self):
        """Update target network with weights from main network."""
        self.target_network.set_weights(self.main_network.get_weights())
    
    def encode_state(self, state):
        """
        Encode a state as a vector for input to the neural network.
        
        Args:
            state (tuple): State as (row, col)
            
        Returns:
            numpy.ndarray: Encoded state vector
        """
        # For this implementation, we'll use a one-hot encoding of the state
        # This is a simple approach for grid worlds
        encoded_state = np.zeros((self.state_size,))
        
        # Calculate flat index for the state (row * width + col)
        row, col = state
        flat_index = row * self.env.width + col
        
        # Set the corresponding index to 1
        encoded_state[flat_index] = 1.0
        
        return encoded_state
    
    def epsilon_greedy_action(self, state):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state (tuple): Current state
            
        Returns:
            int: Selected action
        """
        # Get valid actions from the environment
        valid_actions = self.env.get_valid_actions(state)
        
        if np.random.random() < self.epsilon:
            # Random action
            return np.random.choice(valid_actions)
        else:
            # Greedy action based on Q-values
            encoded_state = self.encode_state(state)
            q_values = self.main_network.predict(np.array([encoded_state]), verbose=0)[0]
            
            # Filter Q-values for valid actions only
            valid_q_values = {action: q_values[action] for action in valid_actions}
            
            # Return action with highest Q-value
            return max(valid_q_values, key=valid_q_values.get)
    
    def train(self, episodes, render_freq=None):
        """
        Train the DQN agent.
        
        Args:
            episodes (int): Number of episodes to run
            render_freq (int): Frequency of rendering during training
            
        Returns:
            dict: Training metrics
        """
        start_time = time.time()
        
        # Training loop
        for episode in range(1, episodes+1):
            # Reset environment
            state = self.env.reset()
            total_reward = 0
            losses = []
            q_values_sum = 0
            q_values_count = 0
            
            done = False
            while not done:
                # Select action
                action = self.epsilon_greedy_action(state)
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Store transition in replay buffer
                self.replay_buffer.add(state, action, reward, next_state, done)
                
                # Move to next state
                state = next_state
                total_reward += reward
                
                # Train if we have enough samples
                if len(self.replay_buffer) >= self.batch_size:
                    loss, avg_q = self._replay()
                    losses.append(loss)
                    q_values_sum += avg_q
                    q_values_count += 1
            
            # Update target network periodically
            if episode % self.target_update_freq == 0:
                self.update_target_network()
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Track metrics
            self.metrics['episode_rewards'].append(total_reward)
            self.metrics['episode_lengths'].append(self.env.steps)
            self.metrics['epsilon_values'].append(self.epsilon)
            
            # Track success code
            success_code = 0  # Default: max steps
            if info['is_goal']:
                success_code = 1  # Goal
            elif info['is_trap']:
                success_code = -1  # Trap
            self.metrics['episode_success'].append(success_code)
            
            # Track average loss and Q-values
            if losses:
                self.metrics['losses'].append(np.mean(losses))
            else:
                self.metrics['losses'].append(0)
                
            if q_values_count > 0:
                self.metrics['avg_q_values'].append(q_values_sum / q_values_count)
            else:
                self.metrics['avg_q_values'].append(0)
            
            # Render if needed
            if render_freq and episode % render_freq == 0:
                print(f"Episode {episode}/{episodes}, Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")
                self.env.render()
        
        # Record total training time
        self.metrics['time'] = time.time() - start_time
        
        return self.metrics
    
    def _replay(self):
        """
        Train the network using experience replay.
        
        Returns:
            tuple: (loss, average Q-value)
        """
        # Sample a batch of transitions
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Encode states
        encoded_states = np.array([self.encode_state(state) for state in states])
        encoded_next_states = np.array([self.encode_state(next_state) for next_state in next_states])
        
        # Get current Q-values
        current_q_values = self.main_network.predict(encoded_states, verbose=0)
        
        # Get target Q-values
        target_q_values = self.target_network.predict(encoded_next_states, verbose=0)
        
        # Prepare training data
        X = encoded_states
        y = current_q_values.copy()
        
        # Update Q-values for the actions that were taken
        for i in range(len(states)):
            if dones[i]:
                # If terminal state, only consider immediate reward
                y[i, actions[i]] = rewards[i]
            else:
                # Otherwise, include discounted future rewards
                y[i, actions[i]] = rewards[i] + self.gamma * np.max(target_q_values[i])
        
        # Train the network
        history = self.main_network.fit(X, y, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # Calculate average Q-value
        avg_q = np.mean([y[i, actions[i]] for i in range(len(actions))])
        
        return loss, avg_q
    
    def get_policy(self, epsilon=0.0):
        """
        Extract policy from trained Q-network.
        
        Args:
            epsilon (float): Exploration rate for the policy
        
        Returns:
            dict: Policy mapping states to actions
        """
        policy = {}
        
        # For each state in the environment
        for row in range(self.env.height):
            for col in range(self.env.width):
                state = (row, col)
                
                # Skip walls and terminal states
                if self.env.grid[row, col] == self.env.WALL or \
                   state in self.env.goal_states or \
                   state in self.env.trap_states:
                    continue
                
                # Get valid actions
                valid_actions = self.env.get_valid_actions(state)
                
                # If no valid actions, skip
                if not valid_actions:
                    continue
                
                # With small probability, choose random action
                if np.random.random() < epsilon:
                    policy[state] = np.random.choice(valid_actions)
                else:
                    # Otherwise, choose best action
                    encoded_state = self.encode_state(state)
                    q_values = self.main_network.predict(np.array([encoded_state]), verbose=0)[0]
                    
                    # Filter valid actions
                    valid_q_values = {action: q_values[action] for action in valid_actions}
                    best_action = max(valid_q_values, key=valid_q_values.get)
                    policy[state] = best_action
        
        return policy
    
    def visualize_policy(self, title="DQN Policy"):
        """
        Visualize the learned policy.
        
        Args:
            title (str): Plot title
        """
        # Get the policy
        policy = self.get_policy()
        
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
    
    def visualize_q_values(self, title="DQN Q-Values"):
        """
        Visualize the Q-values as a heatmap.
        
        Args:
            title (str): Plot title
        """
        # Create a 2D grid for the state values
        grid_values = np.zeros((self.env.height, self.env.width))
        
        # Fill in the values
        for row in range(self.env.height):
            for col in range(self.env.width):
                state = (row, col)
                
                # Skip walls and terminal states
                if self.env.grid[row, col] == self.env.WALL:
                    grid_values[row, col] = np.nan
                    continue
                
                # Encode state
                encoded_state = self.encode_state(state)
                q_values = self.main_network.predict(np.array([encoded_state]), verbose=0)[0]
                
                # Get valid actions
                valid_actions = self.env.get_valid_actions(state)
                
                if valid_actions:
                    # Use maximum Q-value among valid actions
                    grid_values[row, col] = max(q_values[action] for action in valid_actions)
                else:
                    # Terminal state
                    grid_values[row, col] = np.max(q_values)
        
        # Use the environment's render method to show values
        self.env.render(show_values=True, values=grid_values)
        plt.title(title)
        plt.show()
    
    def visualize_learning_curve(self, title="DQN Learning Curve"):
        """
        Visualize the learning curve.
        
        Args:
            title (str): Plot title
        """
        plt.figure(figsize=(15, 10))
        
        # Plot episode rewards
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics['episode_rewards'], label="Episode Rewards")
        # Add moving average for smoothing
        window_size = min(10, len(self.metrics['episode_rewards']))
        if window_size > 0:
            moving_avg = np.convolve(self.metrics['episode_rewards'], 
                                   np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(self.metrics['episode_rewards'])), 
                    moving_avg, 'r', label=f"Moving Avg ({window_size})")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Episode Rewards")
        plt.legend()
        plt.grid(True)
        
        # Plot training loss
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics['losses'], label="Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        
        # Plot epsilon decay
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics['epsilon_values'], label="Epsilon")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon Value")
        plt.title("Exploration Rate")
        plt.grid(True)
        
        # Plot average Q-values
        plt.subplot(2, 2, 4)
        plt.plot(self.metrics['avg_q_values'], label="Avg Q-Value")
        plt.xlabel("Episode")
        plt.ylabel("Average Q-Value")
        plt.title("Q-Value Progression")
        plt.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()
    
    def visualize_success_rate(self, window_size=10, title="DQN Success Rate"):
        """
        Visualize the success rate over episodes.
        
        Args:
            window_size (int): Window size for moving average
            title (str): Plot title
        """
        plt.figure(figsize=(10, 6))
        
        # Convert success codes to success rate
        # 1: reached goal, -1: fell into trap, 0: reached max steps
        success_values = np.array(self.metrics['episode_success'])
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

def dqn(environment, hidden_layers=[64, 64], gamma=0.99, epsilon=1.0, episodes=500):
    """
    Run the DQN algorithm on the given environment.
    
    Args:
        environment: The environment to run on
        hidden_layers (list): Hidden layer sizes for the neural network
        gamma (float): Discount factor
        epsilon (float): Initial exploration rate
        episodes (int): Number of episodes to run
        
    Returns:
        tuple: (model, policy, metrics)
    """
    # Calculate state and action space sizes
    state_size = environment.height * environment.width
    action_size = len(environment.get_all_actions())
    
    # Create DQN agent
    dqn_agent = DQN(
        environment=environment,
        state_size=state_size,
        action_size=action_size,
        hidden_layers=hidden_layers,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=10
    )
    
    # Train the agent
    metrics = dqn_agent.train(episodes=episodes, render_freq=None)
    
    # Extract policy
    policy = dqn_agent.get_policy()
    
    return dqn_agent, policy, metrics