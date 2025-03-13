import numpy as np
import matplotlib.pyplot as plt
from GridWorldEnvironment import GridWorldEnvironment

def test_basic_environment():
    """Test basic environment functionality with a simple grid."""
    print("Testing basic environment functionality...")
    
    grid = [
        [0, 0, 0, 0, 0],
        [0, -1, 0, -1, 0],
        [0, 0, 0, 0, 0],
        [0, -1, 0, -1, 0],
        [0, 0, 0, 0, 0]
    ]
    
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
    
    # Test reset
    state = env.reset()
    assert state == start_state, f"Reset failed: {state} != {start_state}"
    
    # Test valid actions
    valid_actions = env.get_valid_actions()
    assert set(valid_actions) == {env.DOWN, env.RIGHT}, f"Invalid actions: {valid_actions}"
    
    # Test step function with valid action
    next_state, reward, done, info = env.step(env.RIGHT)
    assert next_state == (0, 1), f"Step failed: {next_state} != (0, 1)"
    assert reward == -0.1, f"Incorrect reward: {reward} != -0.1"
    assert not done, f"Episode incorrectly marked as done"
    
    # Test step function with invalid action (should stay in place)
    env.reset()
    next_state, reward, done, info = env.step(env.LEFT)
    assert next_state == (0, 0), f"Step with invalid action failed: {next_state} != (0, 0)"
    
    print("Basic environment test passed!")

def test_stochastic_environment():
    """Test stochastic transitions in the environment."""
    print("Testing stochastic transitions...")
    
    grid = np.zeros((5, 5))
    start_state = (2, 2)
    goal_states = [(4, 4)]
    trap_states = []
    
    env = GridWorldEnvironment(
        grid=grid,
        start_state=start_state,
        goal_states=goal_states,
        trap_states=trap_states,
        is_stochastic=True,
        transition_prob=0.8
    )
    
    # Run many steps and check distribution
    env.reset()
    action_counts = {(2, 2): 0, (2, 3): 0, (1, 2): 0, (3, 2): 0, (2, 1): 0}
    action = env.RIGHT  # Try to move right consistently
    
    n_trials = 1000
    for _ in range(n_trials):
        env.reset()
        next_state, _, _, _ = env.step(action)
        action_counts[next_state] = action_counts.get(next_state, 0) + 1
    
    # Check if distribution roughly matches expected probabilities
    right_prob = action_counts[(2, 3)] / n_trials
    up_prob = action_counts[(1, 2)] / n_trials
    down_prob = action_counts[(3, 2)] / n_trials
    
    print(f"Intended direction (right) probability: {right_prob:.4f} (expected ~0.8)")
    print(f"Perpendicular direction (up) probability: {up_prob:.4f} (expected ~0.1)")
    print(f"Perpendicular direction (down) probability: {down_prob:.4f} (expected ~0.1)")
    
    assert 0.75 <= right_prob <= 0.85, f"Unexpected probability distribution for intended action"
    assert 0.05 <= up_prob <= 0.15, f"Unexpected probability distribution for perpendicular action"
    assert 0.05 <= down_prob <= 0.15, f"Unexpected probability distribution for perpendicular action"
    
    print("Stochastic transitions test passed!")

def test_rewards_and_terminal_states():
    """Test reward calculation and terminal state detection."""
    print("Testing rewards and terminal states...")
    
    grid = np.zeros((5, 5))
    start_state = (0, 0)
    goal_states = [(4, 4)]
    trap_states = [(2, 2)]
    
    env = GridWorldEnvironment(
        grid=grid,
        start_state=start_state,
        goal_states=goal_states,
        trap_states=trap_states,
        step_cost=-0.1,
        goal_reward=1.0,
        trap_penalty=-1.0
    )
    
    # Test goal reward
    env.current_state = (4, 3)
    next_state, reward, done, info = env.step(env.RIGHT)
    assert next_state == (4, 4), f"Step to goal failed: {next_state} != (4, 4)"
    assert reward == 1.0, f"Incorrect goal reward: {reward} != 1.0"
    assert done, f"Episode not marked as done at goal"
    
    # Test trap penalty
    env.reset()
    env.current_state = (2, 1)
    next_state, reward, done, info = env.step(env.RIGHT)
    assert next_state == (2, 2), f"Step to trap failed: {next_state} != (2, 2)"
    assert reward == -1.0, f"Incorrect trap penalty: {reward} != -1.0"
    assert done, f"Episode not marked as done at trap"
    
    # Test custom rewards
    custom_rewards = {(1, 1, env.RIGHT): 0.5}
    env = GridWorldEnvironment(
        grid=grid,
        start_state=start_state,
        goal_states=goal_states,
        trap_states=trap_states,
        step_cost=-0.1,
        custom_rewards=custom_rewards
    )
    
    env.current_state = (1, 1)
    next_state, reward, done, info = env.step(env.RIGHT)
    assert reward == 0.5, f"Incorrect custom reward: {reward} != 0.5"
    
    print("Rewards and terminal states test passed!")

def test_run_episode():
    """Test running a complete episode."""
    print("Testing run_episode functionality...")
    
    grid = [
        [0, 0, 0, 0, 0],
        [0, -1, 0, -1, 0],
        [0, 0, 0, 0, 0],
        [0, -1, 0, -1, 0],
        [0, 0, 0, 0, 0]
    ]
    
    start_state = (0, 0)
    goal_states = [(4, 4)]
    trap_states = []
    
    # Define a simple policy that aims for the goal
    def simple_policy(state):
        row, col = state
        target_row, target_col = goal_states[0]
        
        # Move towards the goal
        if row < target_row and (row+1, col) not in [(1, 1), (3, 1), (1, 3), (3, 3)]:
            return GridWorldEnvironment.DOWN
        elif col < target_col and (row, col+1) not in [(1, 1), (3, 1), (1, 3), (3, 3)]:
            return GridWorldEnvironment.RIGHT
        elif row > target_row:
            return GridWorldEnvironment.UP
        elif col > target_col:
            return GridWorldEnvironment.LEFT
        else:
            return np.random.choice([GridWorldEnvironment.RIGHT, GridWorldEnvironment.DOWN])
    
    env = GridWorldEnvironment(
        grid=grid,
        start_state=start_state,
        goal_states=goal_states,
        trap_states=trap_states,
        step_cost=-0.1
    )
    
    # Run an episode with the policy
    total_reward, steps, states = env.run_episode(policy=simple_policy)
    
    # Check if goal was reached
    assert states[-1] == goal_states[0], f"Goal not reached: final state = {states[-1]}"
    
    print(f"Episode completed in {steps} steps with total reward: {total_reward}")
    print("Run episode test passed!")

def test_visualization():
    """Test visualization capabilities."""
    print("Testing visualization...")
    
    grid = [
        [0, 0, 0, 0, 0],
        [0, -1, 0, -1, 0],
        [0, 0, 0, 0, 0],
        [0, -1, 0, -1, 0],
        [0, 0, 0, 0, 0]
    ]
    
    start_state = (0, 0)
    goal_states = [(4, 4)]
    trap_states = [(2, 2)]
    
    env = GridWorldEnvironment(
        grid=grid,
        start_state=start_state,
        goal_states=goal_states,
        trap_states=trap_states
    )
    
    # Run a random episode
    total_reward, steps, states = env.run_episode()
    
    # Test basic rendering
    fig = env.render(mode='rgb_array', agent_history=True)
    plt.close(fig)
    
    # Test animation (just create it without displaying)
    animation = env.animate_episode(interval=100)
    
    print("Visualization test passed!")

def test_complex_environment():
    """Test a more complex environment configuration."""
    print("Testing complex environment...")
    
    # Create a larger grid with more obstacles and multiple goals/traps
    grid = np.zeros((10, 10))
    
    # Add walls
    walls = [(1, 1), (1, 2), (1, 3), (3, 1), (3, 2), (3, 3),
             (5, 5), (5, 6), (5, 7), (6, 5), (7, 5), (8, 5)]
    for r, c in walls:
        grid[r, c] = -1
    
    start_state = (0, 0)
    goal_states = [(9, 9), (0, 9)]  # Multiple goals with different rewards
    trap_states = [(4, 4), (6, 8)]   # Multiple traps
    
    # Add custom rewards
    custom_rewards = {
        (2, 7, GridWorldEnvironment.UP): 0.5,
        (7, 2, GridWorldEnvironment.RIGHT): 0.3
    }
    
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
    
    # Test state transition and rewards
    all_states = env.get_all_states()
    all_actions = env.get_all_actions()
    
    # Check if we can compute transitions and rewards for all state-action pairs
    for state in all_states:
        for action in all_actions:
            valid_actions = env.get_valid_actions(state)
            if action in valid_actions:
                # Test transition probabilities
                trans_probs = env.get_transition_probabilities(state, action)
                assert sum(trans_probs.values()) > 0.99, f"Transition probabilities don't sum to 1: {trans_probs}"
                
                # Test reward function
                for next_state in trans_probs:
                    reward = env.get_reward(state, action, next_state)
                    # Just check that reward calculation doesn't raise errors
    
    # Run an episode
    total_reward, steps, states = env.run_episode()
    assert len(states) > 0, "No states recorded in episode"
    
    print("Complex environment test passed!")

def test_all():
    """Run all tests."""
    test_basic_environment()
    print()
    test_stochastic_environment()
    print()
    test_rewards_and_terminal_states()
    print()
    test_run_episode()
    print()
    test_visualization()
    print()
    test_complex_environment()
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_all()