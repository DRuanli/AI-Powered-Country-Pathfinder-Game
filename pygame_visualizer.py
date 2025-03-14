import pygame
import numpy as np
import time
import sys

class GridWorldVisualizer:
    """Pygame-based visualizer for the GridWorld environment."""
    
    # Define colors
    COLORS = {
        'background': (240, 240, 240),
        'empty': (255, 255, 255),
        'wall': (40, 40, 40),
        'start': (64, 224, 208),  # Turquoise
        'goal': (50, 205, 50),    # Lime Green
        'trap': (255, 99, 71),    # Tomato Red
        'agent': (0, 0, 255),     # Blue
        'path': (0, 191, 255),    # Deep Sky Blue
        'text': (0, 0, 0),        # Black
        'grid_line': (200, 200, 200),
        'policy_arrow': (255, 0, 0)  # Red
    }
    
    # Define action directions
    ACTION_DIRS = {
        0: (0, -1),  # UP (row-1, col)
        1: (1, 0),   # RIGHT (row, col+1)
        2: (0, 1),   # DOWN (row+1, col)
        3: (-1, 0)   # LEFT (row, col-1)
    }
    
    def __init__(self, env, cell_size=60, padding=10, info_width=300):
        """
        Initialize the visualizer.
        
        Args:
            env: The GridWorldEnvironment
            cell_size: Size of each grid cell in pixels
            padding: Padding around the grid in pixels
            info_width: Width of the info panel in pixels
        """
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # Store environment
        self.env = env
        self.grid = env.grid
        self.height, self.width = env.height, env.width
        
        # Calculate window dimensions
        self.cell_size = cell_size
        self.padding = padding
        self.info_width = info_width
        self.window_width = self.width * cell_size + 2 * padding + info_width
        self.window_height = self.height * cell_size + 2 * padding
        
        # Create window
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Reinforcement Learning Grid World")
        
        # Create fonts
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)
        self.info_font = pygame.font.SysFont('Arial', 16)
        self.cell_font = pygame.font.SysFont('Arial', 12)
        
        # Animation settings
        self.animation_delay = 500  # milliseconds
        self.running = True
        self.episode_running = False
        self.clock = pygame.time.Clock()
        
        # Episode data
        self.agent_position = None
        self.path = []
        self.total_reward = 0
        self.steps = 0
        self.episode_result = None
        
        # Stored policy for visualization
        self.policy = None
        self.action_names = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
    
    def draw_grid(self):
        """Draw the grid and its elements."""
        # Draw background
        self.window.fill(self.COLORS['background'])
        
        # Draw grid cells
        for r in range(self.height):
            for c in range(self.width):
                x = self.padding + c * self.cell_size
                y = self.padding + r * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                
                # Determine cell color
                cell_value = self.grid[r, c]
                
                if (r, c) == self.env.start_state:
                    color = self.COLORS['start']
                elif (r, c) in self.env.goal_states:
                    color = self.COLORS['goal']
                elif (r, c) in self.env.trap_states:
                    color = self.COLORS['trap']
                elif cell_value == -100:  # Wall
                    color = self.COLORS['wall']
                else:
                    color = self.COLORS['empty']
                
                # Draw cell
                pygame.draw.rect(self.window, color, rect)
                pygame.draw.rect(self.window, self.COLORS['grid_line'], rect, 1)
                
                # Add cell text
                if (r, c) == self.env.start_state:
                    text = self.cell_font.render("START", True, self.COLORS['text'])
                elif (r, c) in self.env.goal_states:
                    text = self.cell_font.render("GOAL", True, self.COLORS['text'])
                elif (r, c) in self.env.trap_states:
                    text = self.cell_font.render("TRAP", True, self.COLORS['text'])
                elif cell_value == -100:  # Wall
                    text = self.cell_font.render("", True, self.COLORS['text'])
                else:
                    text = self.cell_font.render("", True, self.COLORS['text'])
                
                text_rect = text.get_rect(center=(x + self.cell_size/2, y + self.cell_size/2))
                self.window.blit(text, text_rect)
                
                # Draw policy arrow if available
                if self.policy is not None:
                    # Skip walls and terminal states
                    if cell_value == -100 or (r, c) in self.env.goal_states or (r, c) in self.env.trap_states:
                        continue
                    
                    # Get action from policy
                    if isinstance(self.policy, dict):
                        if (r, c) in self.policy:
                            action = self.policy[(r, c)]
                        else:
                            continue
                    else:  # Policy is a numpy array
                        if 0 <= r < self.policy.shape[0] and 0 <= c < self.policy.shape[1]:
                            action = self.policy[r, c]
                        else:
                            continue
                    
                    # Draw arrow
                    arrow_color = self.COLORS['policy_arrow']
                    center_x = x + self.cell_size/2
                    center_y = y + self.cell_size/2
                    
                    # Calculate arrow points based on action
                    if action == 0:  # UP
                        pygame.draw.line(self.window, arrow_color, 
                                        (center_x, center_y + 10), 
                                        (center_x, center_y - 10), 2)
                        pygame.draw.polygon(self.window, arrow_color, 
                                          [(center_x - 5, center_y - 5), 
                                           (center_x + 5, center_y - 5),
                                           (center_x, center_y - 15)])
                    elif action == 1:  # RIGHT
                        pygame.draw.line(self.window, arrow_color, 
                                        (center_x - 10, center_y), 
                                        (center_x + 10, center_y), 2)
                        pygame.draw.polygon(self.window, arrow_color, 
                                          [(center_x + 5, center_y - 5), 
                                           (center_x + 5, center_y + 5),
                                           (center_x + 15, center_y)])
                    elif action == 2:  # DOWN
                        pygame.draw.line(self.window, arrow_color, 
                                        (center_x, center_y - 10), 
                                        (center_x, center_y + 10), 2)
                        pygame.draw.polygon(self.window, arrow_color, 
                                          [(center_x - 5, center_y + 5), 
                                           (center_x + 5, center_y + 5),
                                           (center_x, center_y + 15)])
                    elif action == 3:  # LEFT
                        pygame.draw.line(self.window, arrow_color, 
                                        (center_x + 10, center_y), 
                                        (center_x - 10, center_y), 2)
                        pygame.draw.polygon(self.window, arrow_color, 
                                          [(center_x - 5, center_y - 5), 
                                           (center_x - 5, center_y + 5),
                                           (center_x - 15, center_y)])
        
        # Draw agent and path
        if self.agent_position:
            # Draw path first (agent will be on top)
            for i, pos in enumerate(self.path):
                if i > 0:  # Skip the first position (already have the agent)
                    r, c = pos
                    x = self.padding + c * self.cell_size + self.cell_size/2
                    y = self.padding + r * self.cell_size + self.cell_size/2
                    pygame.draw.circle(self.window, self.COLORS['path'], (x, y), self.cell_size/10)
            
            # Draw agent
            r, c = self.agent_position
            x = self.padding + c * self.cell_size + self.cell_size/2
            y = self.padding + r * self.cell_size + self.cell_size/2
            pygame.draw.circle(self.window, self.COLORS['agent'], (x, y), self.cell_size/4)
    
    def draw_info_panel(self):
        """Draw the information panel on the right side."""
        # Panel background
        panel_rect = pygame.Rect(
            self.padding + self.width * self.cell_size + self.padding,
            self.padding,
            self.info_width - self.padding,
            self.height * self.cell_size
        )
        pygame.draw.rect(self.window, (220, 220, 220), panel_rect)
        
        # Title
        title = self.title_font.render("Grid World", True, self.COLORS['text'])
        title_rect = title.get_rect(center=(panel_rect.centerx, self.padding + 20))
        self.window.blit(title, title_rect)
        
        # Current information
        y_offset = self.padding + 60
        
        # Status
        if self.episode_running:
            status_text = "Status: Running"
        elif self.episode_result:
            status_text = f"Status: {self.episode_result}"
        else:
            status_text = "Status: Ready"
        
        status = self.info_font.render(status_text, True, self.COLORS['text'])
        self.window.blit(status, (panel_rect.left + 10, y_offset))
        y_offset += 30
        
        # Steps
        steps_text = self.info_font.render(f"Steps: {self.steps}", True, self.COLORS['text'])
        self.window.blit(steps_text, (panel_rect.left + 10, y_offset))
        y_offset += 30
        
        # Total reward
        reward_text = self.info_font.render(f"Total Reward: {self.total_reward:.2f}", True, self.COLORS['text'])
        self.window.blit(reward_text, (panel_rect.left + 10, y_offset))
        y_offset += 30
        
        # Current position
        if self.agent_position:
            pos_text = self.info_font.render(f"Position: {self.agent_position}", True, self.COLORS['text'])
            self.window.blit(pos_text, (panel_rect.left + 10, y_offset))
        y_offset += 30
        
        # Controls
        y_offset += 20
        controls_title = self.info_font.render("Controls:", True, self.COLORS['text'])
        self.window.blit(controls_title, (panel_rect.left + 10, y_offset))
        y_offset += 25
        
        controls = [
            "SPACE: Start/Pause",
            "R: Reset",
            "+/-: Speed Up/Down",
            "Q: Quit"
        ]
        
        for control in controls:
            control_text = self.info_font.render(control, True, self.COLORS['text'])
            self.window.blit(control_text, (panel_rect.left + 20, y_offset))
            y_offset += 25
        
        # Speed information
        y_offset += 10
        speed_text = self.info_font.render(f"Animation Delay: {self.animation_delay} ms", True, self.COLORS['text'])
        self.window.blit(speed_text, (panel_rect.left + 10, y_offset))
    
    def set_policy(self, policy):
        """Set a policy to visualize."""
        self.policy = policy
    
    def run_episode(self, policy=None):
        """Run an episode with the given policy."""
        if policy:
            self.set_policy(policy)
        
        # Reset environment and state
        state = self.env.reset()
        self.agent_position = state
        self.path = [state]
        self.total_reward = 0
        self.steps = 0
        self.episode_result = None
        self.episode_running = True
        
        # Main loop
        last_action_time = 0
        
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    return
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.running = False
                        pygame.quit()
                        return
                    elif event.key == pygame.K_r:
                        # Reset episode
                        return self.run_episode(policy)
                    elif event.key == pygame.K_SPACE:
                        # Toggle pause
                        self.episode_running = not self.episode_running
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        # Speed up
                        self.animation_delay = max(50, self.animation_delay - 50)
                    elif event.key == pygame.K_MINUS:
                        # Slow down
                        self.animation_delay = min(2000, self.animation_delay + 50)
            
            # Draw current state
            self.draw_grid()
            self.draw_info_panel()
            pygame.display.update()
            
            # If paused, continue to next frame
            if not self.episode_running:
                self.clock.tick(30)
                continue
            
            # Check if it's time for a new action
            current_time = pygame.time.get_ticks()
            if current_time - last_action_time < self.animation_delay:
                self.clock.tick(30)
                continue
            
            # Take action if there's a policy
            if policy and not self.episode_result:
                # Choose action based on policy
                if isinstance(policy, dict):
                    if state in policy:
                        action = policy[state]
                    else:
                        action = np.random.choice(self.env.get_valid_actions(state))
                else:  # Policy is a numpy array
                    r, c = state
                    if 0 <= r < policy.shape[0] and 0 <= c < policy.shape[1]:
                        action = policy[r, c]
                    else:
                        action = np.random.choice(self.env.get_valid_actions(state))
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Update state
                state = next_state
                self.agent_position = state
                self.path.append(state)
                self.total_reward += reward
                self.steps += 1
                last_action_time = current_time
                
                # Check if episode is done
                if done:
                    if info.get('is_goal', False):
                        self.episode_result = "Goal Reached!"
                    elif info.get('is_trap', False):
                        self.episode_result = "Fell into Trap!"
                    else:
                        self.episode_result = "Episode Ended"
            
            # Limit frame rate
            self.clock.tick(30)
    
    def run_interactive(self):
        """Run in interactive mode where the user can control the agent."""
        # Reset environment and state
        state = self.env.reset()
        self.agent_position = state
        self.path = [state]
        self.total_reward = 0
        self.steps = 0
        self.episode_result = None
        
        # Map keys to actions
        key_to_action = {
            pygame.K_UP: 0,    # UP
            pygame.K_RIGHT: 1, # RIGHT
            pygame.K_DOWN: 2,  # DOWN
            pygame.K_LEFT: 3   # LEFT
        }
        
        # Main loop
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    return
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.running = False
                        pygame.quit()
                        return
                    elif event.key == pygame.K_r:
                        # Reset episode
                        return self.run_interactive()
                    elif event.key in key_to_action and not self.episode_result:
                        # Take action
                        action = key_to_action[event.key]
                        
                        # Check if the action is valid
                        valid_actions = self.env.get_valid_actions(state)
                        if action in valid_actions:
                            next_state, reward, done, info = self.env.step(action)
                            
                            # Update state
                            state = next_state
                            self.agent_position = state
                            self.path.append(state)
                            self.total_reward += reward
                            self.steps += 1
                            
                            # Check if episode is done
                            if done:
                                if info.get('is_goal', False):
                                    self.episode_result = "Goal Reached!"
                                elif info.get('is_trap', False):
                                    self.episode_result = "Fell into Trap!"
                                else:
                                    self.episode_result = "Episode Ended"
            
            # Draw current state
            self.draw_grid()
            self.draw_info_panel()
            pygame.display.update()
            
            # Limit frame rate
            self.clock.tick(30)

def visualize_with_ai(env, policy=None, cell_size=60):
    """
    Visualize the environment with AI control.
    
    Args:
        env: The GridWorldEnvironment
        policy: The policy to follow (optional)
        cell_size: Size of grid cells in pixels
    """
    # Create visualizer
    visualizer = GridWorldVisualizer(env, cell_size=cell_size)
    
    # Run episode with policy
    if policy:
        visualizer.run_episode(policy)
    else:
        visualizer.run_interactive()

class LargeMapVisualizer(GridWorldVisualizer):
    """
    Visualizer for large grid world maps with camera controls.
    """
    def __init__(self, env, cell_size=20, view_width=800, view_height=600):
        """
        Initialize the large map visualizer.
        
        Args:
            env: The GridWorldEnvironment
            cell_size: Initial size of each grid cell in pixels
            view_width: Width of the viewport in pixels
            view_height: Height of the viewport in pixels
        """
        # Set dimensions based on view size
        self.view_width = view_width
        self.view_height = view_height
        
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # Store environment
        self.env = env
        self.grid = env.grid
        self.height, self.width = env.height, env.width
        
        # Calculate dimensions
        self.cell_size = cell_size
        self.padding = 10
        self.info_width = 300
        self.window_width = view_width + self.info_width
        self.window_height = view_height
        
        # Create window
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption(f"Large Grid World ({self.width}x{self.height})")
        
        # Create fonts
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)
        self.info_font = pygame.font.SysFont('Arial', 16)
        self.cell_font = pygame.font.SysFont('Arial', 12)
        
        # Animation settings
        self.animation_delay = 500  # milliseconds
        self.running = True
        self.episode_running = False
        self.clock = pygame.time.Clock()
        
        # Episode data
        self.agent_position = None
        self.path = []
        self.total_reward = 0
        self.steps = 0
        self.episode_result = None
        
        # Stored policy for visualization
        self.policy = None
        self.action_names = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
        
        # Camera settings
        self.camera_x = 0
        self.camera_y = 0
        self.min_cell_size = 10
        self.max_cell_size = 100
        
        # Initialize with agent position in view
        self.center_camera_on_agent()
    
    def center_camera_on_agent(self):
        """Center the camera on the agent's position."""
        if self.agent_position:
            self.camera_x = (self.agent_position[1] * self.cell_size) - (self.view_width // 2)
            self.camera_y = (self.agent_position[0] * self.cell_size) - (self.view_height // 2)
        else:
            # Center on start position if agent not placed yet
            self.camera_x = (self.env.start_state[1] * self.cell_size) - (self.view_width // 2)
            self.camera_y = (self.env.start_state[0] * self.cell_size) - (self.view_height // 2)
        
        # Ensure camera stays within bounds
        self.clamp_camera()
    
    def clamp_camera(self):
        """Keep camera within the grid bounds."""
        max_x = self.width * self.cell_size - self.view_width
        max_y = self.height * self.cell_size - self.view_height
        
        self.camera_x = max(0, min(self.camera_x, max_x))
        self.camera_y = max(0, min(self.camera_y, max_y))
    
    def world_to_screen(self, row, col):
        """Convert world coordinates to screen coordinates."""
        screen_x = col * self.cell_size - self.camera_x
        screen_y = row * self.cell_size - self.camera_y
        return screen_x, screen_y
    
    def screen_to_world(self, screen_x, screen_y):
        """Convert screen coordinates to world coordinates."""
        world_col = (screen_x + self.camera_x) // self.cell_size
        world_row = (screen_y + self.camera_y) // self.cell_size
        return int(world_row), int(world_col)
    
    def is_visible(self, row, col):
        """Check if a cell is visible in the viewport."""
        screen_x, screen_y = self.world_to_screen(row, col)
        return (0 <= screen_x <= self.view_width and 
                0 <= screen_y <= self.view_height or
                0 <= screen_x + self.cell_size <= self.view_width and 
                0 <= screen_y + self.cell_size <= self.view_height)
    
    def draw_grid(self):
        """Draw the visible portion of the grid."""
        # Draw background
        self.window.fill(self.COLORS['background'])
        
        # Calculate visible range
        start_row, start_col = self.screen_to_world(0, 0)
        end_row, end_col = self.screen_to_world(self.view_width, self.view_height)
        
        # Add padding to ensure all partially visible cells are drawn
        start_row = max(0, start_row - 1)
        start_col = max(0, start_col - 1)
        end_row = min(self.height, end_row + 2)
        end_col = min(self.width, end_col + 2)
        
        # Draw visible grid cells
        for r in range(start_row, end_row):
            for c in range(start_col, end_col):
                screen_x, screen_y = self.world_to_screen(r, c)
                
                # Skip if not on screen
                if screen_x + self.cell_size < 0 or screen_y + self.cell_size < 0:
                    continue
                if screen_x > self.view_width or screen_y > self.view_height:
                    continue
                
                rect = pygame.Rect(screen_x, screen_y, self.cell_size, self.cell_size)
                
                # Determine cell color
                cell_value = self.grid[r, c]
                
                if (r, c) == self.env.start_state:
                    color = self.COLORS['start']
                elif (r, c) in self.env.goal_states:
                    color = self.COLORS['goal']
                elif (r, c) in self.env.trap_states:
                    color = self.COLORS['trap']
                elif cell_value == -100:  # Wall
                    color = self.COLORS['wall']
                else:
                    color = self.COLORS['empty']
                
                # Draw cell
                pygame.draw.rect(self.window, color, rect)
                pygame.draw.rect(self.window, self.COLORS['grid_line'], rect, 1)
                
                # Add cell text if cells are large enough
                if self.cell_size >= 30:
                    if (r, c) == self.env.start_state:
                        text = self.cell_font.render("START", True, self.COLORS['text'])
                    elif (r, c) in self.env.goal_states:
                        text = self.cell_font.render("GOAL", True, self.COLORS['text'])
                    elif (r, c) in self.env.trap_states:
                        text = self.cell_font.render("TRAP", True, self.COLORS['text'])
                    else:
                        continue
                    
                    text_rect = text.get_rect(center=(screen_x + self.cell_size/2, screen_y + self.cell_size/2))
                    self.window.blit(text, text_rect)
                
                # Draw policy arrow if available and cell is large enough
                if self.policy is not None and self.cell_size >= 20:
                    # Skip walls and terminal states
                    if cell_value == -100 or (r, c) in self.env.goal_states or (r, c) in self.env.trap_states:
                        continue
                    
                    # Get action from policy
                    if isinstance(self.policy, dict):
                        if (r, c) in self.policy:
                            action = self.policy[(r, c)]
                        else:
                            continue
                    else:  # Policy is a numpy array
                        if 0 <= r < self.policy.shape[0] and 0 <= c < self.policy.shape[1]:
                            action = self.policy[r, c]
                        else:
                            continue
                    
                    # Draw arrow
                    arrow_color = self.COLORS['policy_arrow']
                    center_x = screen_x + self.cell_size/2
                    center_y = screen_y + self.cell_size/2
                    
                    # Calculate arrow size relative to cell size
                    arrow_size = max(5, self.cell_size / 6)
                    
                    # Calculate arrow points based on action
                    if action == 0:  # UP
                        pygame.draw.line(self.window, arrow_color, 
                                        (center_x, center_y + arrow_size), 
                                        (center_x, center_y - arrow_size), 2)
                        pygame.draw.polygon(self.window, arrow_color, 
                                          [(center_x - arrow_size/2, center_y - arrow_size/2), 
                                           (center_x + arrow_size/2, center_y - arrow_size/2),
                                           (center_x, center_y - arrow_size*1.5)])
                    elif action == 1:  # RIGHT
                        pygame.draw.line(self.window, arrow_color, 
                                        (center_x - arrow_size, center_y), 
                                        (center_x + arrow_size, center_y), 2)
                        pygame.draw.polygon(self.window, arrow_color, 
                                          [(center_x + arrow_size/2, center_y - arrow_size/2), 
                                           (center_x + arrow_size/2, center_y + arrow_size/2),
                                           (center_x + arrow_size*1.5, center_y)])
                    elif action == 2:  # DOWN
                        pygame.draw.line(self.window, arrow_color, 
                                        (center_x, center_y - arrow_size), 
                                        (center_x, center_y + arrow_size), 2)
                        pygame.draw.polygon(self.window, arrow_color, 
                                          [(center_x - arrow_size/2, center_y + arrow_size/2), 
                                           (center_x + arrow_size/2, center_y + arrow_size/2),
                                           (center_x, center_y + arrow_size*1.5)])
                    elif action == 3:  # LEFT
                        pygame.draw.line(self.window, arrow_color, 
                                        (center_x + arrow_size, center_y), 
                                        (center_x - arrow_size, center_y), 2)
                        pygame.draw.polygon(self.window, arrow_color, 
                                          [(center_x - arrow_size/2, center_y - arrow_size/2), 
                                           (center_x - arrow_size/2, center_y + arrow_size/2),
                                           (center_x - arrow_size*1.5, center_y)])
        
        # Draw agent and path if visible
        if self.agent_position:
            # Draw path first (agent will be on top)
            for i, pos in enumerate(self.path):
                if i > 0:  # Skip the first position (already have the agent)
                    r, c = pos
                    screen_x, screen_y = self.world_to_screen(r, c)
                    
                    # Only draw if on screen
                    if 0 <= screen_x <= self.view_width and 0 <= screen_y <= self.view_height:
                        pygame.draw.circle(self.window, self.COLORS['path'], 
                                          (screen_x + self.cell_size/2, screen_y + self.cell_size/2), 
                                          max(2, self.cell_size/10))
            
            # Draw agent
            r, c = self.agent_position
            screen_x, screen_y = self.world_to_screen(r, c)
            
            # Only draw if on screen
            if 0 <= screen_x <= self.view_width and 0 <= screen_y <= self.view_height:
                pygame.draw.circle(self.window, self.COLORS['agent'], 
                                  (screen_x + self.cell_size/2, screen_y + self.cell_size/2), 
                                  max(5, self.cell_size/4))
        
        # Draw viewport border
        pygame.draw.rect(self.window, (0, 0, 0), 
                         pygame.Rect(0, 0, self.view_width, self.view_height), 2)
    
    def draw_info_panel(self):
        """Draw the information panel on the right side."""
        # Panel background
        panel_rect = pygame.Rect(
            self.view_width,
            0,
            self.info_width,
            self.window_height
        )
        pygame.draw.rect(self.window, (220, 220, 220), panel_rect)
        
        # Title
        title = self.title_font.render("Large Grid World", True, self.COLORS['text'])
        title_rect = title.get_rect(center=(panel_rect.centerx, 20))
        self.window.blit(title, title_rect)
        
        # Grid info
        y_offset = 60
        grid_info = self.info_font.render(f"Grid: {self.width}x{self.height}", True, self.COLORS['text'])
        self.window.blit(grid_info, (panel_rect.left + 10, y_offset))
        y_offset += 25
        
        # Current information
        if self.agent_position:
            pos_text = self.info_font.render(f"Position: {self.agent_position}", True, self.COLORS['text'])
            self.window.blit(pos_text, (panel_rect.left + 10, y_offset))
        y_offset += 25
        
        # Status
        if self.episode_running:
            status_text = "Status: Running"
        elif self.episode_result:
            status_text = f"Status: {self.episode_result}"
        else:
            status_text = "Status: Ready"
        
        status = self.info_font.render(status_text, True, self.COLORS['text'])
        self.window.blit(status, (panel_rect.left + 10, y_offset))
        y_offset += 25
        
        # Steps
        steps_text = self.info_font.render(f"Steps: {self.steps}", True, self.COLORS['text'])
        self.window.blit(steps_text, (panel_rect.left + 10, y_offset))
        y_offset += 25
        
        # Total reward
        reward_text = self.info_font.render(f"Total Reward: {self.total_reward:.2f}", True, self.COLORS['text'])
        self.window.blit(reward_text, (panel_rect.left + 10, y_offset))
        y_offset += 25
        
        # Camera info
        y_offset += 10
        zoom_text = self.info_font.render(f"Zoom: {self.cell_size}px/cell", True, self.COLORS['text'])
        self.window.blit(zoom_text, (panel_rect.left + 10, y_offset))
        y_offset += 25
        
        # Controls
        y_offset += 20
        controls_title = self.info_font.render("Controls:", True, self.COLORS['text'])
        self.window.blit(controls_title, (panel_rect.left + 10, y_offset))
        y_offset += 25
        
        controls = [
            "SPACE: Start/Pause",
            "R: Reset",
            "+/-: Speed Up/Down",
            "Q: Quit",
            "WASD: Move Camera",
            "Mouse Wheel: Zoom",
            "C: Center on Agent"
        ]
        
        for control in controls:
            control_text = self.info_font.render(control, True, self.COLORS['text'])
            self.window.blit(control_text, (panel_rect.left + 20, y_offset))
            y_offset += 25
        
        # Speed information
        y_offset += 10
        speed_text = self.info_font.render(f"Animation Delay: {self.animation_delay} ms", True, self.COLORS['text'])
        self.window.blit(speed_text, (panel_rect.left + 10, y_offset))
    
    def handle_camera_controls(self):
        """Handle camera movement and zoom controls."""
        keys = pygame.key.get_pressed()
        
        # Camera movement
        camera_speed = max(10, self.cell_size)
        if keys[pygame.K_w]:
            self.camera_y -= camera_speed
        if keys[pygame.K_s]:
            self.camera_y += camera_speed
        if keys[pygame.K_a]:
            self.camera_x -= camera_speed
        if keys[pygame.K_d]:
            self.camera_x += camera_speed
        
        # Keep camera in bounds
        self.clamp_camera()
    
    def run_episode(self, policy=None):
        """Run an episode with the given policy."""
        if policy:
            self.set_policy(policy)
        
        # Reset environment and state
        state = self.env.reset()
        self.agent_position = state
        self.path = [state]
        self.total_reward = 0
        self.steps = 0
        self.episode_result = None
        self.episode_running = True
        
        # Center camera on agent
        self.center_camera_on_agent()
        
        # Main loop
        last_action_time = 0
        
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    return
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.running = False
                        pygame.quit()
                        return
                    elif event.key == pygame.K_r:
                        # Reset episode
                        return self.run_episode(policy)
                    elif event.key == pygame.K_SPACE:
                        # Toggle pause
                        self.episode_running = not self.episode_running
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        # Speed up
                        self.animation_delay = max(50, self.animation_delay - 50)
                    elif event.key == pygame.K_MINUS:
                        # Slow down
                        self.animation_delay = min(2000, self.animation_delay + 50)
                    elif event.key == pygame.K_c:
                        # Center on agent
                        self.center_camera_on_agent()
                
                # Handle zoom with mouse wheel
                if event.type == pygame.MOUSEWHEEL:
                    # Zoom in/out
                    zoom_factor = 1.1
                    if event.y > 0:  # Scroll up = zoom in
                        self.cell_size = min(self.max_cell_size, self.cell_size * zoom_factor)
                    elif event.y < 0:  # Scroll down = zoom out
                        self.cell_size = max(self.min_cell_size, self.cell_size / zoom_factor)
                    
                    # Adjust camera to maintain zoom center
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    if mouse_x < self.view_width and mouse_y < self.view_height:
                        # Get world position before zoom
                        world_row_before, world_col_before = self.screen_to_world(mouse_x, mouse_y)
                        
                        # Update camera position to keep the same world position under mouse
                        world_x_after = world_col_before * self.cell_size
                        world_y_after = world_row_before * self.cell_size
                        self.camera_x = world_x_after - mouse_x
                        self.camera_y = world_y_after - mouse_y
                        
                        # Keep camera in bounds
                        self.clamp_camera()
            
            # Handle continuous camera movement
            self.handle_camera_controls()
            
            # Draw current state
            self.draw_grid()
            self.draw_info_panel()
            pygame.display.update()
            
            # If paused, continue to next frame
            if not self.episode_running:
                self.clock.tick(30)
                continue
            
            # Check if it's time for a new action
            current_time = pygame.time.get_ticks()
            if current_time - last_action_time < self.animation_delay:
                self.clock.tick(30)
                continue
            
            # Take action if there's a policy
            if policy and not self.episode_result:
                # Choose action based on policy
                if isinstance(policy, dict):
                    if state in policy:
                        action = policy[state]
                    else:
                        action = np.random.choice(self.env.get_valid_actions(state))
                else:  # Policy is a numpy array
                    r, c = state
                    if 0 <= r < policy.shape[0] and 0 <= c < policy.shape[1]:
                        action = policy[r, c]
                    else:
                        action = np.random.choice(self.env.get_valid_actions(state))
                
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Update state
                state = next_state
                self.agent_position = state
                self.path.append(state)
                self.total_reward += reward
                self.steps += 1
                last_action_time = current_time
                
                # Check if agent is visible, if not center camera
                agent_screen_x, agent_screen_y = self.world_to_screen(state[0], state[1])
                if not (0 <= agent_screen_x <= self.view_width and 0 <= agent_screen_y <= self.view_height):
                    self.center_camera_on_agent()
                
                # Check if episode is done
                if done:
                    if info.get('is_goal', False):
                        self.episode_result = "Goal Reached!"
                    elif info.get('is_trap', False):
                        self.episode_result = "Fell into Trap!"
                    else:
                        self.episode_result = "Episode Ended"
            
            # Limit frame rate
            self.clock.tick(30)
    
    def run_interactive(self):
        """Run in interactive mode where the user can control the agent."""
        # Reset environment and state
        state = self.env.reset()
        self.agent_position = state
        self.path = [state]
        self.total_reward = 0
        self.steps = 0
        self.episode_result = None
        
        # Center camera on agent
        self.center_camera_on_agent()
        
        # Map keys to actions
        action_keys = {
            pygame.K_UP: 0,    # UP
            pygame.K_RIGHT: 1, # RIGHT
            pygame.K_DOWN: 2,  # DOWN
            pygame.K_LEFT: 3   # LEFT
        }
        
        # Main loop
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.quit()
                    return
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.running = False
                        pygame.quit()
                        return
                    elif event.key == pygame.K_r:
                        # Reset episode
                        return self.run_interactive()
                    elif event.key == pygame.K_c:
                        # Center on agent
                        self.center_camera_on_agent()
                    elif event.key in action_keys and not self.episode_result:
                        # Take action
                        action = action_keys[event.key]
                        
                        # Check if the action is valid
                        valid_actions = self.env.get_valid_actions(state)
                        if action in valid_actions:
                            next_state, reward, done, info = self.env.step(action)
                            
                            # Update state
                            state = next_state
                            self.agent_position = state
                            self.path.append(state)
                            self.total_reward += reward
                            self.steps += 1
                            
                            # Check if episode is done
                            if done:
                                if info.get('is_goal', False):
                                    self.episode_result = "Goal Reached!"
                                elif info.get('is_trap', False):
                                    self.episode_result = "Fell into Trap!"
                                else:
                                    self.episode_result = "Episode Ended"
                
                # Handle zoom with mouse wheel
                if event.type == pygame.MOUSEWHEEL:
                    # Zoom in/out
                    zoom_factor = 1.1
                    if event.y > 0:  # Scroll up = zoom in
                        self.cell_size = min(self.max_cell_size, self.cell_size * zoom_factor)
                    elif event.y < 0:  # Scroll down = zoom out
                        self.cell_size = max(self.min_cell_size, self.cell_size / zoom_factor)
                    
                    # Adjust camera to maintain zoom center
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    if mouse_x < self.view_width and mouse_y < self.view_height:
                        # Get world position before zoom
                        world_row_before, world_col_before = self.screen_to_world(mouse_x, mouse_y)
                        
                        # Update camera position to keep the same world position under mouse
                        world_x_after = world_col_before * self.cell_size
                        world_y_after = world_row_before * self.cell_size
                        self.camera_x = world_x_after - mouse_x
                        self.camera_y = world_y_after - mouse_y
                        
                        # Keep camera in bounds
                        self.clamp_camera()
            
            # Handle continuous camera movement
            self.handle_camera_controls()
            
            # Draw current state
            self.draw_grid()
            self.draw_info_panel()
            pygame.display.update()
            
            # Limit frame rate
            self.clock.tick(30)

def visualize_interactive(env, cell_size=60):
    """
    Visualize the environment with user control.
    
    Args:
        env: The GridWorldEnvironment
        cell_size: Size of grid cells in pixels
    """
    # Create visualizer
    visualizer = GridWorldVisualizer(env, cell_size=cell_size)
    
    # Run interactive mode
    visualizer.run_interactive()