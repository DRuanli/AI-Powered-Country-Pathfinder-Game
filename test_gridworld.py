from GridWorldEnvironment import GridWorldEnvironment

# Test just the trap reward calculation
def test_trap_reward():
    grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    start_state = (0, 0)
    goal_states = [(2, 2)]
    trap_states = [(1, 1)]
    
    env = GridWorldEnvironment(
        grid=grid,
        start_state=start_state,
        goal_states=goal_states,
        trap_states=trap_states,
        trap_penalty=1.0  # This should be interpreted as a positive value
    )
    
    # Move to trap
    env.current_state = (1, 0)
    next_state, reward, done, info = env.step(env.RIGHT)
    
    print(f"Moving to trap state {next_state}")
    print(f"Reward received: {reward}")
    print(f"Expected reward: -1.0")
    print(f"Is trap state: {next_state in trap_states}")
    print(f"Done flag: {done}")

if __name__ == "__main__":
    test_trap_reward()