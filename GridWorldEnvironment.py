import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from IPython.display import display, HTML
import time

class GridWorldEnvironment:
    """
    A flexible grid world environment for reinforcement learning.
    
    This environment supports:
    - Variable grid dimensions
    - Different cell types (empty, walls, goals, traps)
    - Deterministic or stochastic transitions
    - Customizable reward structure
    """
    
    # Define action constants
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    # Cell type constants
    EMPTY = 0
    WALL = -1
    
    # Action direction mappings
    ACTION_DIRS = {
        UP: (-1, 0),
        RIGHT: (0, 1),
        DOWN: (1, 0),
        LEFT: (0, -1)
    }
    
    # Action names for visualization
    ACTION_NAMES = {
        UP: "↑",
        RIGHT: "→",
        DOWN: "↓",
        LEFT: "←"
    }
    
    def __init__(self, grid, start_state, goal_states, trap_states, 
                 is_stochastic=False, transition_prob=0.8, 
                 step_cost=-0.01, goal_reward=1.0, trap_penalty=1.0,
                 custom_rewards=None, max_steps=1000):
        """
        Initialize the environment with the given parameters.
        
        Args:
            grid (np.ndarray or list): The grid layout. 0 for empty cells, -1 for walls,
                                      positive values for goals, negative values for traps.
            start_state (tuple): The starting position (row, col).
            goal_states (list): List of goal positions [(row, col), ...].
            trap_states (list): List of trap positions [(row, col), ...].
            is_stochastic (bool): If True, actions have a probability of moving in a different direction.
            transition_prob (float): Probability of moving in the intended direction (if stochastic).
            step_cost (float): Small cost for each step to encourage efficiency.
            goal_reward (float): Reward for reaching a goal state.
            trap_penalty (float): Penalty for reaching a trap state.
            custom_rewards (dict): Optional custom rewards for specific state-action pairs.
                                  Format: {(row, col, action): reward}
            max_steps (int): Maximum number of steps per episode.
        """
        # Convert grid to numpy array if it's a list
        self.grid = np.array(grid, dtype=float)
        self.height, self.width = self.grid.shape
        
        # Validate start, goal, and trap states
        self._validate_state(start_state, "start_state")
        for goal in goal_states:
            self._validate_state(goal, "goal_state")
        for trap in trap_states:
            self._validate_state(trap, "trap_state")
        
        # Set environment parameters
        self.start_state = start_state
        self.goal_states = goal_states
        self.trap_states = trap_states
        self.is_stochastic = is_stochastic
        self.transition_prob = transition_prob
        
        # Set reward parameters
        self.step_cost = step_cost
        self.goal_reward = goal_reward
        self.trap_penalty = trap_penalty
        self.custom_rewards = custom_rewards or {}
        
        # Set grid cell values for goals and traps
        # If not already set, we use the provided rewards
        for goal in goal_states:
            if self.grid[goal] <= 0:
                self.grid[goal] = goal_reward
                
        for trap in trap_states:
            if self.grid[trap] >= 0:
                self.grid[trap] = -trap_penalty  # Note: grid shows negative value for traps
        
        # Initialize environment state
        self.current_state = None
        self.steps = 0
        self.max_steps = max_steps
        self.episode_rewards = []
        self.episode_states = []
        
        # Initialize the environment
        self.reset()
        
    def _validate_state(self, state, state_name):
        """Validate that a state is within the grid boundaries and not a wall."""
        row, col = state
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise ValueError(f"{state_name} {state} is outside grid boundaries. Grid dimensions: {self.height}x{self.width}")
        if self.grid[row, col] == self.WALL:
            raise ValueError(f"{state_name} {state} is inside a wall. Please ensure this position isn't marked as a wall in your grid.")
    
    def reset(self):
        """
        Reset the environment to the initial state.
        
        Returns:
            tuple: The initial state (row, col).
        """
        self.current_state = self.start_state
        self.steps = 0
        self.episode_rewards = []
        self.episode_states = [self.current_state]
        return self.current_state
    
    def step(self, action):
        """
        Take a step in the environment based on the given action.
        
        Args:
            action (int): The action to take (UP=0, RIGHT=1, DOWN=2, LEFT=3).
        
        Returns:
            tuple: (next_state, reward, done, info)
                next_state: The new state after taking the action.
                reward: The reward received.
                done: Whether the episode is finished (reached goal or max steps).
                info: Additional information (e.g., step count).
        """
        if action not in self.ACTION_DIRS:
            raise ValueError(f"Invalid action: {action}. Must be 0-3.")
        
        # Increment step counter
        self.steps += 1
        
        # Determine next state based on action and stochasticity
        next_state = self._get_next_state(self.current_state, action)
        
        # Calculate reward
        reward = self._calculate_reward(self.current_state, action, next_state)
        
        # Track state and reward
        self.current_state = next_state
        self.episode_rewards.append(reward)
        self.episode_states.append(next_state)
        
        # Check if episode is done
        in_goal = any(next_state == goal for goal in self.goal_states)
        in_trap = any(next_state == trap for trap in self.trap_states)
        done = in_goal or in_trap or self.steps >= self.max_steps
        
        # Additional info
        info = {
            'steps': self.steps,
            'is_goal': in_goal,
            'is_trap': in_trap,
            'is_max_steps': self.steps >= self.max_steps
        }
        
        return next_state, reward, done, info
    
    def _get_next_state(self, state, action):
        """
        Determine the next state given current state and action.
        Accounts for stochasticity if is_stochastic=True.
        """
        if self.is_stochastic:
            # With probability transition_prob, move as intended
            # With probability (1-transition_prob), move in one of the perpendicular directions
            rand = np.random.random()
            if rand < self.transition_prob:
                intended_action = action
            else:
                # Get perpendicular actions
                perp_actions = [(action + 1) % 4, (action - 1) % 4]
                intended_action = np.random.choice(perp_actions)
        else:
            intended_action = action
        
        # Get direction of movement
        dr, dc = self.ACTION_DIRS[intended_action]
        row, col = state
        new_row, new_col = row + dr, col + dc
        
        # Check if new position is valid (within grid and not a wall)
        if (0 <= new_row < self.height and 
            0 <= new_col < self.width and 
            self.grid[new_row, new_col] != self.WALL):
            return (new_row, new_col)
        else:
            # If invalid, stay in the same position
            return state
    
    def _calculate_reward(self, state, action, next_state):
        """Calculate the reward for a state-action-next_state transition."""
        # Check for custom reward
        if (state[0], state[1], action) in self.custom_rewards:
            return self.custom_rewards[(state[0], state[1], action)]
        
        # Check if next state is a goal
        for goal in self.goal_states:
            if next_state == goal:
                return self.goal_reward
        
        # Check if next state is a trap
        for trap in self.trap_states:
            if next_state == trap:
                return -self.trap_penalty  # Negative because it's a penalty
        
        # Otherwise, return step cost for taking an action
        return self.step_cost
    
    def _is_terminal_state(self, state):
        """Check if the state is terminal (goal or trap)."""
        # Explicitly check membership to ensure correct behavior
        for goal in self.goal_states:
            if state == goal:
                return True
        for trap in self.trap_states:
            if state == trap:
                return True
        return False
    
    def get_valid_actions(self, state=None):
        """
        Return the list of valid actions from the current or specified state.
        
        Args:
            state (tuple, optional): The state to check. Defaults to current state.
        
        Returns:
            list: List of valid actions (UP=0, RIGHT=1, DOWN=2, LEFT=3).
        """
        if state is None:
            state = self.current_state
            
        valid_actions = []
        row, col = state
        
        for action, (dr, dc) in self.ACTION_DIRS.items():
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.height and 
                0 <= new_col < self.width and 
                self.grid[new_row, new_col] != self.WALL):
                valid_actions.append(action)
                
        return valid_actions
    
    def render(self, mode='human', show_values=False, values=None, policy=None, agent_history=False):
        """
        Visualize the current state of the environment.
        
        Args:
            mode (str): The mode for rendering ('human' or 'rgb_array').
            show_values (bool): Whether to show state values on the grid.
            values (dict or np.ndarray): Values to display for each state.
            policy (dict or np.ndarray): Policy (actions) to display for each state.
            agent_history (bool): Whether to show the agent's path history.
            
        Returns:
            fig: The matplotlib figure object if mode='rgb_array', else None.
        """
        # Create a colored grid for visualization
        grid_values = np.copy(self.grid)
        
        # Create a colormap for the grid
        cmap = plt.cm.RdYlGn  # Red for negative, yellow for neutral, green for positive
        norm = mcolors.TwoSlopeNorm(vmin=min(min(grid_values.flatten()), -0.1), 
                                   vcenter=0,
                                   vmax=max(max(grid_values.flatten()), 0.1))
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Show the grid
        mesh = ax.pcolormesh(grid_values.T, cmap=cmap, norm=norm, edgecolors='k', linewidth=1)
        ax.invert_yaxis()  # Invert y-axis to match row, col coordinates
        
        # Add colorbar
        plt.colorbar(mesh, ax=ax, label='Cell Value')
        
        # Mark special cells
        ax.scatter([self.start_state[0] + 0.5], [self.start_state[1] + 0.5], 
                  color='blue', marker='o', s=100, label='Start')
        
        for goal in self.goal_states:
            ax.scatter([goal[0] + 0.5], [goal[1] + 0.5], 
                      color='green', marker='*', s=150, label='Goal')
        
        for trap in self.trap_states:
            ax.scatter([trap[0] + 0.5], [trap[1] + 0.5], 
                      color='red', marker='x', s=100, label='Trap')
        
        # Mark walls
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == self.WALL:
                    ax.add_patch(plt.Rectangle((i, j), 1, 1, color='black'))
        
        # Show state values if requested
        if show_values and values is not None:
            if isinstance(values, dict):
                for (r, c), val in values.items():
                    if self.grid[r, c] != self.WALL:
                        ax.text(r + 0.5, c + 0.5, f"{val:.2f}", 
                               ha='center', va='center', color='black')
            elif isinstance(values, np.ndarray):
                for r in range(self.height):
                    for c in range(self.width):
                        if self.grid[r, c] != self.WALL:
                            ax.text(r + 0.5, c + 0.5, f"{values[r, c]:.2f}", 
                                   ha='center', va='center', color='black')
        
        # Show policy if provided
        if policy is not None:
            if isinstance(policy, dict):
                for (r, c), action in policy.items():
                    if self.grid[r, c] != self.WALL:
                        ax.text(r + 0.5, c + 0.5, self.ACTION_NAMES[action], 
                               ha='center', va='center', color='blue', fontsize=15)
            elif isinstance(policy, np.ndarray):
                for r in range(self.height):
                    for c in range(self.width):
                        if self.grid[r, c] != self.WALL and (r, c) not in self.goal_states and (r, c) not in self.trap_states:
                            ax.text(r + 0.5, c + 0.5, self.ACTION_NAMES[policy[r, c]], 
                                   ha='center', va='center', color='blue', fontsize=15)
        
        # Mark agent's current position
        ax.scatter([self.current_state[0] + 0.5], [self.current_state[1] + 0.5], 
                  color='purple', marker='o', s=150, label='Agent')
        
        # Show agent's history if requested
        if agent_history and len(self.episode_states) > 1:
            path_rows, path_cols = zip(*self.episode_states)
            ax.plot([r + 0.5 for r in path_rows], [c + 0.5 for c in path_cols], 
                   'purple', alpha=0.5, linewidth=2)
        
        # Add grid labels
        ax.set_xticks(np.arange(0.5, self.height + 0.5))
        ax.set_yticks(np.arange(0.5, self.width + 0.5))
        ax.set_xticklabels(range(self.height))
        ax.set_yticklabels(range(self.width))
        ax.set_xlabel('Row')
        ax.set_ylabel('Column')
        
        # Add title
        ax.set_title('GridWorld Environment')
        
        # Handle legend duplicates
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        if mode == 'human':
            plt.show()
            return None
        elif mode == 'rgb_array':
            return fig
    
    def animate_episode(self, episode=None, interval=500, save_path=None):
        """
        Visualize an entire episode (sequence of states and actions).
        
        Args:
            episode (list, optional): List of states to animate. Defaults to current episode.
            interval (int): Interval between frames in milliseconds.
            save_path (str, optional): Path to save the animation. If None, display only.
            
        Returns:
            HTML: Animation object that can be displayed in Jupyter notebook.
        """
        if episode is None:
            episode = self.episode_states
            
        if len(episode) < 2:
            print("Not enough states to animate. Run an episode first.")
            return None
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create a function to update the plot for each frame
        def update(frame):
            ax.clear()
            
            # Show the grid
            grid_values = np.copy(self.grid)
            cmap = plt.cm.RdYlGn
            norm = mcolors.TwoSlopeNorm(vmin=min(min(grid_values.flatten()), -0.1), 
                                       vcenter=0,
                                       vmax=max(max(grid_values.flatten()), 0.1))
            
            mesh = ax.pcolormesh(grid_values.T, cmap=cmap, norm=norm, edgecolors='k', linewidth=1)
            ax.invert_yaxis()
            
            # Mark special cells
            ax.scatter([self.start_state[0] + 0.5], [self.start_state[1] + 0.5], 
                      color='blue', marker='o', s=100, label='Start')
            
            for goal in self.goal_states:
                ax.scatter([goal[0] + 0.5], [goal[1] + 0.5], 
                          color='green', marker='*', s=150, label='Goal')
            
            for trap in self.trap_states:
                ax.scatter([trap[0] + 0.5], [trap[1] + 0.5], 
                          color='red', marker='x', s=100, label='Trap')
            
            # Mark walls
            for i in range(self.height):
                for j in range(self.width):
                    if self.grid[i, j] == self.WALL:
                        ax.add_patch(plt.Rectangle((i, j), 1, 1, color='black'))
            
            # Mark agent's current position
            current_pos = episode[frame]
            ax.scatter([current_pos[0] + 0.5], [current_pos[1] + 0.5], 
                      color='purple', marker='o', s=150, label='Agent')
            
            # Show agent's path up to this point
            if frame > 0:
                path = episode[:frame+1]
                path_rows, path_cols = zip(*path)
                ax.plot([r + 0.5 for r in path_rows], [c + 0.5 for c in path_cols], 
                       'purple', alpha=0.5, linewidth=2)
            
            # Add grid labels
            ax.set_xticks(np.arange(0.5, self.height + 0.5))
            ax.set_yticks(np.arange(0.5, self.width + 0.5))
            ax.set_xticklabels(range(self.height))
            ax.set_yticklabels(range(self.width))
            ax.set_xlabel('Row')
            ax.set_ylabel('Column')
            
            # Add title with frame information
            ax.set_title(f'GridWorld Episode - Step {frame}')
            
            # Handle legend duplicates
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))
            
            return [ax]
        
        # Create the animation
        ani = FuncAnimation(fig, update, frames=range(len(episode)), 
                            interval=interval, blit=False)
        
        # Save if path is provided
        if save_path:
            ani.save(save_path, writer='pillow')
            
        plt.close(fig)
        
        # Display in notebook
        if not save_path:
            return HTML(ani.to_jshtml())
        else:
            return None
    
    def run_episode(self, policy=None, max_steps=None):
        """
        Run an episode using a given policy or random actions.
        
        Args:
            policy (dict or function): Policy to follow. If dict, maps states to actions.
                If function, takes state and returns action. If None, uses random actions.
            max_steps (int, optional): Maximum steps to take. Defaults to self.max_steps.
            
        Returns:
            tuple: (total_reward, steps, states)
                total_reward: Sum of rewards in the episode.
                steps: Number of steps taken.
                states: List of states visited.
        """
        if max_steps is None:
            max_steps = self.max_steps
            
        state = self.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Determine action based on policy
            if policy is None:
                action = np.random.choice(self.get_valid_actions(state))
            elif callable(policy):
                action = policy(state)
            else:
                action = policy.get(state, np.random.choice(self.get_valid_actions(state)))
            
            # Take step
            next_state, reward, done, _ = self.step(action)
            total_reward += reward
            state = next_state
            steps += 1
        
        return total_reward, steps, self.episode_states

    def get_transition_probabilities(self, state, action):
        """
        Get the transition probabilities for a state-action pair.
        
        Args:
            state (tuple): The state (row, col).
            action (int): The action (UP=0, RIGHT=1, DOWN=2, LEFT=3).
            
        Returns:
            dict: Dictionary mapping next_states to probabilities.
        """
        if not self.is_stochastic:
            # Deterministic case
            next_state = self._get_next_state(state, action)
            return {next_state: 1.0}
        
        # Stochastic case
        probs = {}
        
        # Intended action with probability transition_prob
        intended_next_state = self._get_next_state(state, action)
        probs[intended_next_state] = self.transition_prob
        
        # Perpendicular actions with probability (1-transition_prob)/2 each
        perp_prob = (1 - self.transition_prob) / 2
        perp_actions = [(action + 1) % 4, (action - 1) % 4]
        
        for perp_action in perp_actions:
            perp_next_state = self._get_next_state(state, perp_action)
            if perp_next_state in probs:
                probs[perp_next_state] += perp_prob
            else:
                probs[perp_next_state] = perp_prob
        
        return probs
    
    def get_reward(self, state, action, next_state):
        """
        Get the reward for a state-action-next_state transition.
        
        Args:
            state (tuple): The current state (row, col).
            action (int): The action taken (UP=0, RIGHT=1, DOWN=2, LEFT=3).
            next_state (tuple): The next state (row, col).
            
        Returns:
            float: The reward for this transition.
        """
        return self._calculate_reward(state, action, next_state)
    
    def get_all_states(self):
        """Get a list of all possible states (excluding walls)."""
        states = []
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r, c] != self.WALL:
                    states.append((r, c))
        return states
    
    def get_all_actions(self):
        """Get a list of all possible actions."""
        return list(self.ACTION_DIRS.keys())


# Example usage:
if __name__ == "__main__":
    # Create a simple 5x5 grid
    grid = [
        [0, 0, 0, 0, 0],
        [0, -1, 0, -1, 0],
        [0, 0, 0, 0, 0],
        [0, -1, 0, -1, 0],
        [0, 0, 0, 0, 0]
    ]
    
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
        is_stochastic=True,
        transition_prob=0.8,
        step_cost=-0.1
    )
    
    # Run a random episode
    total_reward, steps, states = env.run_episode()
    print(f"Episode completed in {steps} steps with total reward: {total_reward}")
    
    # Visualize the final state
    env.render(agent_history=True)
    
    # Animate the episode
    animation = env.animate_episode()
    display(animation)