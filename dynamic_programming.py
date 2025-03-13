import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
from matplotlib.patches import Arrow, FancyArrowPatch
from GridWorldEnvironment import GridWorldEnvironment

class DynamicProgramming:
    """
    Dynamic Programming methods for reinforcement learning.
    
    Implements Value Iteration and Policy Iteration algorithms for finding
    optimal policies in environments with known dynamics.
    """
    
    def __init__(self, environment):
        """
        Initialize DynamicProgramming with an environment.
        
        Args:
            environment: A GridWorldEnvironment instance
        """
        self.env = environment
        
    def value_iteration(self, gamma=0.9, epsilon=1e-6, max_iterations=1000):
        """
        Implement Value Iteration algorithm.
        
        Args:
            gamma (float): Discount factor
            epsilon (float): Convergence threshold
            max_iterations (int): Maximum number of iterations
            
        Returns:
            tuple: (policy, value_function, metrics)
                policy: 2D array mapping states to actions
                value_function: 2D array of state values
                metrics: Dictionary containing performance metrics
        """
        # Initialize metrics
        metrics = {
            'iterations': 0,
            'time': 0,
            'value_changes': []
        }
        
        # Get grid dimensions
        height, width = self.env.height, self.env.width
        
        # Initialize value function
        V = np.zeros((height, width))
        
        # Set walls to None to avoid calculations
        for r in range(height):
            for c in range(width):
                if self.env.grid[r, c] == self.env.WALL:
                    V[r, c] = None
        
        # Get all possible states and actions
        all_states = self.env.get_all_states()
        all_actions = self.env.get_all_actions()
        
        # Start timer
        start_time = time.time()
        
        # Value iteration loop
        for i in range(max_iterations):
            delta = 0
            V_new = np.copy(V)
            
            # Update value for each state
            for state in all_states:
                r, c = state
                
                # Skip terminal states (goals and traps)
                if state in self.env.goal_states or state in self.env.trap_states:
                    continue
                
                # Calculate values for all actions
                action_values = []
                
                for action in all_actions:
                    # Get transition probabilities
                    transitions = self.env.get_transition_probabilities(state, action)
                    
                    # Calculate expected value
                    expected_value = 0
                    for next_state, prob in transitions.items():
                        # Get reward
                        reward = self.env.get_reward(state, action, next_state)
                        
                        # Get next state value (handling terminal states)
                        next_r, next_c = next_state
                        
                        if next_state in self.env.goal_states:
                            # For goal states, just use reward (no future rewards)
                            next_value = 0
                        elif next_state in self.env.trap_states:
                            # For trap states, just use reward (no future rewards)
                            next_value = 0
                        else:
                            next_value = V[next_r, next_c]
                        
                        expected_value += prob * (reward + gamma * next_value)
                    
                    action_values.append(expected_value)
                
                # Update value with maximum expected value
                if action_values:
                    V_new[r, c] = max(action_values)
                    delta = max(delta, abs(V_new[r, c] - V[r, c]))
            
            # Update value function
            V = V_new
            
            # Track value changes
            metrics['value_changes'].append(delta)
            metrics['iterations'] += 1
            
            # Check for convergence
            if delta < epsilon:
                break
        
        # Record elapsed time
        metrics['time'] = time.time() - start_time
        
        # Extract policy from value function
        policy = np.zeros((height, width), dtype=int)
        
        for state in all_states:
            r, c = state
            
            # Skip walls, goals, and traps
            if self.env.grid[r, c] == self.env.WALL or state in self.env.goal_states or state in self.env.trap_states:
                continue
            
            # Find best action
            best_action = None
            best_value = float('-inf')
            
            for action in all_actions:
                # Get transition probabilities
                transitions = self.env.get_transition_probabilities(state, action)
                
                # Calculate expected value
                expected_value = 0
                for next_state, prob in transitions.items():
                    # Get reward
                    reward = self.env.get_reward(state, action, next_state)
                    
                    # Get next state value
                    next_r, next_c = next_state
                    
                    if next_state in self.env.goal_states:
                        next_value = 0
                    elif next_state in self.env.trap_states:
                        next_value = 0
                    else:
                        next_value = V[next_r, next_c]
                    
                    expected_value += prob * (reward + gamma * next_value)
                
                if expected_value > best_value:
                    best_value = expected_value
                    best_action = action
            
            if best_action is not None:
                policy[r, c] = best_action
        
        return policy, V, metrics
    
    def policy_iteration(self, gamma=0.9, max_iterations=100, eval_epsilon=1e-6, eval_max_iterations=100):
        """
        Implement Policy Iteration algorithm.
        
        Args:
            gamma (float): Discount factor
            max_iterations (int): Maximum number of policy iterations
            eval_epsilon (float): Convergence threshold for policy evaluation
            eval_max_iterations (int): Maximum iterations for policy evaluation
            
        Returns:
            tuple: (policy, value_function, metrics)
                policy: 2D array mapping states to actions
                value_function: 2D array of state values
                metrics: Dictionary containing performance metrics
        """
        # Initialize metrics
        metrics = {
            'iterations': 0,
            'time': 0,
            'policy_changes': []
        }
        
        # Get grid dimensions
        height, width = self.env.height, self.env.width
        
        # Initialize policy randomly
        policy = np.zeros((height, width), dtype=int)
        for r in range(height):
            for c in range(width):
                if (r, c) in self.env.goal_states or (r, c) in self.env.trap_states or self.env.grid[r, c] == self.env.WALL:
                    continue
                valid_actions = self.env.get_valid_actions((r, c))
                if valid_actions:
                    policy[r, c] = np.random.choice(valid_actions)
        
        # Initialize value function
        V = np.zeros((height, width))
        
        # Set walls to None to avoid calculations
        for r in range(height):
            for c in range(width):
                if self.env.grid[r, c] == self.env.WALL:
                    V[r, c] = None
        
        # Get all possible states and actions
        all_states = self.env.get_all_states()
        all_actions = self.env.get_all_actions()
        
        # Start timer
        start_time = time.time()
        
        # Policy iteration loop
        for i in range(max_iterations):
            # 1. Policy Evaluation
            V = self._policy_evaluation(policy, V, gamma, eval_epsilon, eval_max_iterations)
            
            # 2. Policy Improvement
            policy_stable = True
            policy_changes = 0
            
            for state in all_states:
                r, c = state
                
                # Skip walls, goals, and traps
                if self.env.grid[r, c] == self.env.WALL or state in self.env.goal_states or state in self.env.trap_states:
                    continue
                
                old_action = policy[r, c]
                
                # Find best action
                best_action = None
                best_value = float('-inf')
                
                for action in all_actions:
                    # Get transition probabilities
                    transitions = self.env.get_transition_probabilities(state, action)
                    
                    # Calculate expected value
                    expected_value = 0
                    for next_state, prob in transitions.items():
                        # Get reward
                        reward = self.env.get_reward(state, action, next_state)
                        
                        # Get next state value
                        next_r, next_c = next_state
                        
                        if next_state in self.env.goal_states:
                            next_value = 0
                        elif next_state in self.env.trap_states:
                            next_value = 0
                        else:
                            next_value = V[next_r, next_c]
                        
                        expected_value += prob * (reward + gamma * next_value)
                    
                    if expected_value > best_value:
                        best_value = expected_value
                        best_action = action
                
                if best_action is not None:
                    policy[r, c] = best_action
                
                # Check if policy changed
                if old_action != policy[r, c]:
                    policy_stable = False
                    policy_changes += 1
            
            # Record metrics
            metrics['policy_changes'].append(policy_changes)
            metrics['iterations'] += 1
            
            # Check if policy is stable
            if policy_stable:
                break
        
        # Record elapsed time
        metrics['time'] = time.time() - start_time
        
        return policy, V, metrics
    
    def _policy_evaluation(self, policy, V_init, gamma, epsilon, max_iterations):
        """
        Evaluate a policy by computing its value function.
        
        Args:
            policy (numpy.ndarray): Current policy
            V_init (numpy.ndarray): Initial value function
            gamma (float): Discount factor
            epsilon (float): Convergence threshold
            max_iterations (int): Maximum number of iterations
            
        Returns:
            numpy.ndarray: Value function for the given policy
        """
        # Initialize value function from input
        V = np.copy(V_init)
        
        # Get all states
        all_states = self.env.get_all_states()
        
        # Policy evaluation loop
        for i in range(max_iterations):
            delta = 0
            
            for state in all_states:
                r, c = state
                
                # Skip walls, goals, and traps
                if self.env.grid[r, c] == self.env.WALL or state in self.env.goal_states or state in self.env.trap_states:
                    continue
                
                # Get action from policy
                action = policy[r, c]
                
                # Get transition probabilities
                transitions = self.env.get_transition_probabilities(state, action)
                
                # Calculate expected value
                expected_value = 0
                for next_state, prob in transitions.items():
                    # Get reward
                    reward = self.env.get_reward(state, action, next_state)
                    
                    # Get next state value
                    next_r, next_c = next_state
                    
                    if next_state in self.env.goal_states:
                        next_value = 0
                    elif next_state in self.env.trap_states:
                        next_value = 0
                    else:
                        next_value = V[next_r, next_c]
                    
                    expected_value += prob * (reward + gamma * next_value)
                
                # Track max change
                old_value = V[r, c]
                V[r, c] = expected_value
                delta = max(delta, abs(old_value - V[r, c]))
            
            # Check for convergence
            if delta < epsilon:
                break
        
        return V
    
    def visualize_value_function(self, value_function, title="Value Function"):
        """
        Visualize the value function as a heatmap.
        
        Args:
            value_function (numpy.ndarray): Value function to visualize
            title (str): Plot title
        """
        # Create a copy for visualization
        values_for_plot = np.copy(value_function)
        
        # Replace None with NaN for visualization
        for r in range(self.env.height):
            for c in range(self.env.width):
                if values_for_plot[r, c] is None:
                    values_for_plot[r, c] = np.nan
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create colormap
        cmap = plt.cm.viridis
        norm = mcolors.Normalize(vmin=np.nanmin(values_for_plot), vmax=np.nanmax(values_for_plot))
        
        # Create heatmap
        plt.imshow(values_for_plot, cmap=cmap, norm=norm)
        
        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label("Value")
        
        # Add labels
        plt.title(title)
        plt.xlabel("Column")
        plt.ylabel("Row")
        
        # Add grid
        plt.grid(False)
        
        # Add value annotations
        for r in range(self.env.height):
            for c in range(self.env.width):
                if self.env.grid[r, c] == self.env.WALL:
                    plt.text(c, r, "WALL", ha="center", va="center", color="white")
                elif (r, c) in self.env.goal_states:
                    plt.text(c, r, "GOAL", ha="center", va="center", color="white")
                elif (r, c) in self.env.trap_states:
                    plt.text(c, r, "TRAP", ha="center", va="center", color="white")
                elif not np.isnan(values_for_plot[r, c]):
                    plt.text(c, r, f"{values_for_plot[r, c]:.2f}", ha="center", va="center", color="black")
        
        plt.tight_layout()
        plt.show()
    
    def visualize_policy(self, policy, value_function=None, title="Policy"):
        """
        Visualize the policy as arrows on a grid.
        
        Args:
            policy (numpy.ndarray): Policy to visualize
            value_function (numpy.ndarray, optional): Value function to use as background
            title (str): Plot title
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create background using value function if provided
        if value_function is not None:
            # Create a copy for visualization
            values_for_plot = np.copy(value_function)
            
            # Replace None with NaN for visualization
            for r in range(self.env.height):
                for c in range(self.env.width):
                    if values_for_plot[r, c] is None:
                        values_for_plot[r, c] = np.nan
            
            # Create colormap
            cmap = plt.cm.viridis
            norm = mcolors.Normalize(vmin=np.nanmin(values_for_plot), vmax=np.nanmax(values_for_plot))
            
            # Create heatmap
            im = ax.imshow(values_for_plot, cmap=cmap, norm=norm)
            
            # Add colorbar
            cbar = fig.colorbar(im)
            cbar.set_label("Value")
        else:
            # Just create a simple grid
            ax.imshow(np.zeros_like(policy), cmap="binary", alpha=0.1)
        
        # Define arrow directions
        directions = {
            self.env.UP: (0, -0.5),
            self.env.RIGHT: (0.5, 0),
            self.env.DOWN: (0, 0.5),
            self.env.LEFT: (-0.5, 0)
        }
        
        # Plot arrows for each state
        for r in range(self.env.height):
            for c in range(self.env.width):
                if self.env.grid[r, c] == self.env.WALL:
                    ax.text(c, r, "WALL", ha="center", va="center", color="white")
                elif (r, c) in self.env.goal_states:
                    ax.text(c, r, "GOAL", ha="center", va="center", color="white")
                elif (r, c) in self.env.trap_states:
                    ax.text(c, r, "TRAP", ha="center", va="center", color="white")
                else:
                    action = policy[r, c]
                    dx, dy = directions[action]
                    
                    # Create arrow
                    ax.add_patch(FancyArrowPatch((c, r), (c + dx, r + dy),
                                              arrowstyle="->",
                                              color="red",
                                              mutation_scale=15,
                                              linewidth=2))
        
        # Add labels
        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        
        # Set ticks
        ax.set_xticks(np.arange(self.env.width))
        ax.set_yticks(np.arange(self.env.height))
        
        plt.tight_layout()
        plt.show()
    
    def visualize_convergence(self, metrics, algorithm="Value Iteration"):
        """
        Visualize the convergence of the algorithm.
        
        Args:
            metrics (dict): Metrics from the algorithm
            algorithm (str): Name of the algorithm
        """
        plt.figure(figsize=(10, 6))
        
        if algorithm == "Value Iteration":
            plt.plot(metrics["value_changes"], marker="o")
            plt.ylabel("Max Value Change")
            plt.title(f"Value Iteration Convergence ({metrics['iterations']} iterations, {metrics['time']:.3f} seconds)")
        else:  # Policy Iteration
            plt.plot(metrics["policy_changes"], marker="o")
            plt.ylabel("Number of Policy Changes")
            plt.title(f"Policy Iteration Convergence ({metrics['iterations']} iterations, {metrics['time']:.3f} seconds)")
        
        plt.xlabel("Iteration")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def compare_algorithms(self, value_metrics, policy_metrics):
        """
        Compare the performance of Value Iteration and Policy Iteration.
        
        Args:
            value_metrics (dict): Metrics from Value Iteration
            policy_metrics (dict): Metrics from Policy Iteration
        """
        # Comparison metrics
        metrics = {
            "Iterations": [value_metrics["iterations"], policy_metrics["iterations"]],
            "Time (seconds)": [value_metrics["time"], policy_metrics["time"]]
        }
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot iterations comparison
        axes[0].bar(["Value Iteration", "Policy Iteration"], metrics["Iterations"])
        axes[0].set_ylabel("Number of Iterations")
        axes[0].set_title("Iterations Until Convergence")
        axes[0].grid(axis="y")
        
        # Plot time comparison
        axes[1].bar(["Value Iteration", "Policy Iteration"], metrics["Time (seconds)"])
        axes[1].set_ylabel("Time (seconds)")
        axes[1].set_title("Computation Time")
        axes[1].grid(axis="y")
        
        plt.tight_layout()
        plt.show()
        
        # Print additional information
        print("Performance Comparison:")
        print(f"Value Iteration: {value_metrics['iterations']} iterations, {value_metrics['time']:.6f} seconds")
        print(f"Policy Iteration: {policy_metrics['iterations']} iterations, {policy_metrics['time']:.6f} seconds")
        
        # Efficiency calculation
        value_time_per_iter = value_metrics['time'] / value_metrics['iterations'] if value_metrics['iterations'] > 0 else float('inf')
        policy_time_per_iter = policy_metrics['time'] / policy_metrics['iterations'] if policy_metrics['iterations'] > 0 else float('inf')
        
        print(f"\nTime per iteration:")
        print(f"Value Iteration: {value_time_per_iter:.6f} seconds/iteration")
        print(f"Policy Iteration: {policy_time_per_iter:.6f} seconds/iteration")