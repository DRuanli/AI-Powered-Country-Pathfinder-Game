from GridWorldEnvironment import GridWorldEnvironment
import numpy as np

# Test just the trap reward calculation
def test_trap_reward():
    # Create a simple 3x3 grid
    grid = np.zeros((3, 3))
    
    start_state = (0, 0)
    goal_states = [(2, 2)]
    trap_states = [(1, 1)]
    
    env = GridWorldEnvironment(
        grid=grid,
        start_state=start_state,
        goal_states=goal_states,
        trap_states=trap_states,
        trap_penalty=1.0,  # This should be interpreted as a positive value
        is_stochastic=False  # Make sure movement is deterministic for testing
    )
    
    # Move to trap
    env.current_state = (1, 0)
    print(f"Current state: {env.current_state}")
    print(f"Trap states: {env.trap_states}")
    print(f"Moving RIGHT from (1, 0) should go to (1, 1)")
    
    next_state, reward, done, info = env.step(env.RIGHT)
    
    print(f"Next state: {next_state}")
    print(f"Reward received: {reward}")
    print(f"Expected reward: -1.0")
    print(f"Is trap state: {next_state in env.trap_states}")
    print(f"Done flag: {done}")
    
    # Print debug info about the trap
    print("\nDebug information:")
    for r in range(env.height):
        for c in range(env.width):
            print(f"Grid[{r},{c}] = {env.grid[r,c]}")

if __name__ == "__main__":
    test_trap_reward()