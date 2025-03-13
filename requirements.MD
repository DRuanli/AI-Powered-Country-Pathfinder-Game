# FINAL ASSIGNMENT - REINFORCEMENT LEARNING PROJECT

## Course: Introduction to Artificial Intelligence
## Duration: 06 weeks

### I. Formation and Overview

• The project is conducted in groups of 03 – 05 students.
• Student groups will implement and analyze several reinforcement learning algorithms within custom grid-based environments.
• This project will help you gain hands-on experience with fundamental reinforcement learning concepts, algorithm implementation, and experimental analysis.

### II. Requirements and Detailed Specifications

Given the project folder **ReinforcementLearning** consisting of:

• `gridworld.py`: A source code file providing basic functionality for creating and visualizing grid-based environments. This includes helper functions for rendering grids and calculating distances.
• `agents.py`: A template file with base classes for implementing various learning agents. You will extend these base classes.
• `utils.py`: Utility functions for data processing, visualization, and statistical analysis.

#### Core Reinforcement Learning Concepts

Before starting implementation, ensure you understand these fundamental RL concepts:

- **State**: Representation of the environment at each time step
- **Action**: Choices available to the agent at each state
- **Reward**: Feedback signal from the environment after taking an action
- **Policy**: Strategy for selecting actions in different states
- **Value Function**: Estimate of expected future rewards from a state
- **Q-Function**: Estimate of expected future rewards from a state-action pair
- **Discount Factor**: Parameter that determines the importance of future rewards
- **Exploration vs. Exploitation**: Balance between trying new actions and using known good actions

#### a) Task 1 (2.0 points): Environment and Problem Formulation

The student group will implement a flexible grid world environment that supports various reinforcement learning scenarios. Your environment should be customizable in terms of:

1. **Grid Dimensions**: Allow variable sized grids (e.g., 5×5, 10×10, etc.)
2. **Cell Types**:
   - Empty spaces (value = 0): Agent can move freely
   - Walls (value = -1): Agent cannot move into these cells
   - Goal states (value > 0): Positive rewards when reached
   - Trap states (value < 0): Negative rewards when reached
   - Special cells: Optional custom cell types with specific behaviors

3. **Transition Dynamics**:
   - Deterministic: Actions always result in expected movement
   - Stochastic: Actions have a probability (e.g., 0.8) of moving as intended, and smaller probabilities of moving in other directions
   
4. **Reward Structure**:
   - Goal reward: Positive value when reaching goal state
   - Trap penalty: Negative value when reaching trap state
   - Step cost: Small negative value for each action (encourages efficiency)
   - Custom rewards: Ability to define special rewards for specific state-action pairs

The Environment class must include these core methods:
```python
def __init__(self, grid, start_state, goal_states, trap_states, is_stochastic=False, transition_prob=0.8):
    # Initialize the environment with the given parameters
    
def reset(self):
    # Reset the environment to initial state
    # Returns: initial state
    
def step(self, action):
    # Take a step in the environment based on the given action
    # Returns: next_state, reward, done, info
    
def get_valid_actions(self, state):
    # Return the list of valid actions from the current state
    
def render(self):
    # Visualize the current state of the environment
    
def animate_episode(self, episode):
    # Visualize an entire episode (sequence of states and actions)
```

**Detailed Explanation**: Your environment class serves as the foundation for all reinforcement learning algorithms. It must accurately simulate the dynamics of the grid world, calculate appropriate rewards, and handle edge cases (e.g., attempting to move into walls). The visualization functions should create clear, informative displays of the environment state and agent trajectories.

#### b) Task 2 (2.0 points): Value Iteration and Policy Iteration

Implement the `DynamicProgramming` class with detailed implementations of two fundamental model-based reinforcement learning algorithms. These algorithms assume complete knowledge of the environment dynamics.

**value_iteration(environment, gamma, epsilon)** → policy, value_function

Value iteration works by iteratively updating the value function for each state until convergence:

1. Initialize value function V(s) = 0 for all states
2. Repeat until convergence (max change in values < epsilon):
   a. For each state s:
      i. Calculate the new value: V(s) = max_a [ sum_s' P(s'|s,a) * (R(s,a,s') + gamma * V(s')) ]
   b. Check convergence: If max|V_new(s) - V_old(s)| < epsilon, stop
3. Extract policy: For each state s, pi(s) = argmax_a [ sum_s' P(s'|s,a) * (R(s,a,s') + gamma * V(s')) ]

**policy_iteration(environment, gamma)** → policy, value_function

Policy iteration alternates between policy evaluation and policy improvement:

1. Initialize policy pi(s) randomly and value function V(s) = 0
2. Repeat until policy stabilizes:
   a. Policy Evaluation: Compute V_pi by solving the linear equations:
      V_pi(s) = sum_s' P(s'|s,pi(s)) * (R(s,pi(s),s') + gamma * V_pi(s'))
   b. Policy Improvement: Update policy:
      pi(s) = argmax_a [ sum_s' P(s'|s,a) * (R(s,a,s') + gamma * V(s')) ]
   c. If policy remains unchanged, stop

**Implementation Requirements**:
- Use NumPy for efficient matrix operations
- Implement proper termination conditions to avoid infinite loops
- Create visualizations that show:
  - The value function as a heat map
  - The policy as directional arrows on the grid
  - Convergence metrics (value changes across iterations)
- Compare performance metrics between the two algorithms:
  - Number of iterations until convergence
  - Total computation time
  - Final policy quality

#### c) Task 3 (2.0 points): Q-Learning and SARSA

Implement model-free reinforcement learning algorithms that learn through experience rather than requiring a complete model of the environment.

**q_learning(environment, alpha, gamma, epsilon, episodes)** → q_table, policy

Q-learning is an off-policy TD algorithm that directly learns the optimal Q-function:

1. Initialize Q(s,a) = 0 for all state-action pairs
2. For each episode:
   a. Initialize state s
   b. Repeat until s is terminal:
      i. Choose action a from s using policy derived from Q (e.g., epsilon-greedy)
      ii. Take action a, observe reward r and next state s'
      iii. Update Q(s,a) = Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
      iv. s = s'

**sarsa(environment, alpha, gamma, epsilon, episodes)** → q_table, policy

SARSA is an on-policy TD algorithm that learns the Q-function for the current policy:

1. Initialize Q(s,a) = 0 for all state-action pairs
2. For each episode:
   a. Initialize state s
   b. Choose action a from s using policy derived from Q (e.g., epsilon-greedy)
   c. Repeat until s is terminal:
      i. Take action a, observe reward r and next state s'
      ii. Choose action a' from s' using policy derived from Q
      iii. Update Q(s,a) = Q(s,a) + alpha * [r + gamma * Q(s',a') - Q(s,a)]
      iv. s = s', a = a'

**Detailed Implementation Requirements**:
- Implement epsilon-greedy action selection:
  ```python
  def epsilon_greedy(q_values, state, epsilon):
      # With probability epsilon, choose a random action
      # Otherwise, choose the action with the highest Q-value
  ```
- Track performance metrics during training:
  - Average reward per episode
  - Steps to reach goal
  - Exploration rate over time
- Create learning curve visualizations showing improvement over episodes
- Implement decay schedules for exploration (epsilon) and learning rate (alpha)
- Compare Q-learning and SARSA in terms of:
  - Stability of learning
  - Final policy quality
  - Learning speed
  - Performance in deterministic vs. stochastic environments

#### d) Task 4 (2.0 points): Deep Q-Network (DQN)

Implement a Deep Q-Network that uses neural networks to approximate the Q-function, allowing for better generalization in large state spaces.

**dqn(environment, hidden_layers, gamma, epsilon, episodes)** → model, policy

The DQN algorithm extends Q-learning with neural networks and includes two key improvements:
- Experience replay: Store and randomly sample transitions to break correlations
- Target network: Use a separate network for generating targets to improve stability

Implementation steps:
1. Create two neural networks: main network and target network
2. Initialize replay memory buffer with capacity N
3. For each episode:
   a. Initialize state s
   b. Repeat until s is terminal:
      i. With probability epsilon, select random action; otherwise select a = argmax_a Q(s,a;θ)
      ii. Execute action a, observe reward r and next state s'
      iii. Store transition (s,a,r,s') in replay memory
      iv. Sample random mini-batch of transitions from replay memory
      v. Set target y_j = r_j if episode terminates at step j+1, otherwise r_j + gamma * max_a' Q(s'_j,a';θ-)
      vi. Perform gradient descent step on (y_j - Q(s_j,a_j;θ))² with respect to θ
      vii. Every C steps, update target network parameters: θ- = θ
      viii. s = s'

**Detailed Implementation Requirements**:
- Use TensorFlow or PyTorch for neural network implementation
- Create a replay buffer class:
  ```python
  class ReplayBuffer:
      def __init__(self, capacity):
          # Initialize buffer with fixed capacity
          
      def add(self, state, action, reward, next_state, done):
          # Add transition to buffer, overwriting oldest if at capacity
          
      def sample(self, batch_size):
          # Return random batch of transitions
  ```
- Implement network architecture with appropriate state encoding
- Create a step-by-step training process with regular evaluation
- Visualize:
  - Training loss over time
  - Average Q-values during training
  - Network predictions vs. actual returns
- Track hyperparameter effects:
  - Buffer size impact
  - Target network update frequency
  - Neural network architecture choices
  - Batch size

#### e) Task 5 (2.0 points): Comparative Analysis and Custom Environment

Create advanced test environments and conduct comprehensive comparisons between all implemented algorithms.

**Custom Environment Requirements**:
1. Create at least two custom grid worlds with different challenges:
   - Dynamic environment: Elements that change based on agent actions
   - Multi-goal environment: Multiple objectives with different rewards
   - Partially observable environment: Limited visibility around agent
   - Time-dependent rewards: Rewards that change over time

2. For each environment, define:
   - Clear objectives
   - Reward structure
   - Success criteria
   - Difficulty factors

**Comparative Analysis Requirements**:
1. Evaluate all algorithms (Value Iteration, Policy Iteration, Q-Learning, SARSA, DQN) on:
   - Solution quality: Final cumulative reward over test episodes
   - Learning efficiency: Number of iterations/episodes until convergence
   - Robustness: Performance under stochastic vs. deterministic conditions
   - Parameter sensitivity: How algorithm performance varies with parameter changes

2. Create comparative visualizations:
   - Learning curves on the same graph
   - Bar charts of final performance
   - Heatmaps showing parameter sensitivity
   - Trajectory comparisons in the environment

3. Statistical Analysis:
   - Run multiple trials (at least 10) for each algorithm-environment combination
   - Calculate mean and standard deviation of performance metrics
   - Perform statistical significance tests (e.g., t-tests) to compare algorithms

4. Apply algorithms to a real-world inspired scenario (choose one):
   - Navigation problem based on a simplified city map
   - Resource allocation problem
   - Game-playing scenario
   - Traffic management simulation

#### f) Presentation (2.0 points)

Prepare a comprehensive presentation that effectively communicates your implementation approach, results, and insights.

**Detailed Presentation Requirements**:
* Title slide with project name and group members
* Introduction to reinforcement learning (brief theoretical background)
* Environment explanation with visual examples
* For each algorithm:
  * Theoretical explanation with pseudocode
  * Implementation details and key design choices
  * Performance visualizations
  * Strengths and limitations observed
* Comparative analysis results with clear visualizations
* Discussion of findings:
  * Which algorithms performed best under which conditions?
  * How did environment characteristics affect algorithm performance?
  * What practical insights were gained about algorithm selection?
* Conclusion and future work suggestions
* References in IEEE format

**Presentation Format**:
* 15-20 slides maximum
* 4:3 slide ratio for projector compatibility
* High contrast between text and background for readability
* Consistent visual style throughout
* Minimal text per slide (use visuals where possible)
* Practice to ensure presentation fits within 10-minute time limit

### III. Submission Instructions

- Create a folder named: `RL_<Group ID>`
- Content:
  * `source/` → Source code folder, containing:
    - `environment.py`: Environment implementation
    - `dynamic_programming.py`: Value and Policy Iteration
    - `td_learning.py`: Q-Learning and SARSA
    - `deep_rl.py`: DQN implementation
    - `analysis.py`: Comparative analysis code
    - `test_*.py`: Test files for each algorithm
    - Additional utility files as needed
  * `presentation.pdf` → Final presentation slides
  * `report.pdf` → Detailed technical report (5-10 pages)
  * `demo.txt` → URL to a demonstration video (3 minutes maximum)
  * `requirements.txt` → Python package dependencies
  * `README.md` → Setup and execution instructions

- Compress the folder to a zip file and submit by the deadline.

### IV. Evaluation Criteria

Your project will be evaluated based on the following criteria:

1. **Correctness (40%)**:
   - Algorithms implement the theoretical principles correctly
   - Environment properly handles all edge cases
   - Code executes without errors

2. **Implementation Quality (20%)**:
   - Code is well-organized, modular, and follows OOP principles
   - Documentation is clear and comprehensive
   - Efficient implementation avoiding unnecessary computations

3. **Analysis Depth (20%)**:
   - Thorough comparison between algorithms
   - Insightful observations about performance differences
   - Well-designed experiments and visualizations

4. **Presentation Quality (10%)**:
   - Clear communication of concepts and results
   - Professional slide design and organization
   - Effective use of time during presentation

5. **Creativity and Extensions (10%)**:
   - Novel environment designs
   - Additional algorithm features beyond requirements
   - Creative applications or visualizations

### V. Policy

- Student groups submitting late get 0.0 points for each member.
- Missing required materials in the submission loses at least 50% of presentation points.
- Copying source code from the internet/other students, sharing work with other groups, etc. results in 0.0 points for all related groups.
- If signs of illegal copying or sharing are detected, extra interviews will be conducted to verify student groups' work.
- Each group member must be able to explain any part of the implementation during evaluation.

### VI. Additional Resources

To support your implementation, the following resources are recommended:

1. Textbooks:
   - Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
   - Géron, A. (2019). Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow. O'Reilly Media.

2. Online Courses:
   - David Silver's RL Course (DeepMind)
   - Stanford CS234: Reinforcement Learning

3. Libraries:
   - NumPy for efficient array operations
   - Matplotlib for visualization
   - TensorFlow or PyTorch for neural network implementation
   - OpenAI Gym for environment design inspiration

-- THE END --