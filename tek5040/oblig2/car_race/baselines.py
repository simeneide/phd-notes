import numpy as np
import gym

env = gym.make('CarRacing-v0')

# Random action
def random(observation):
    batch_size = observation.shape[0]
    return [env.action_space.sample() for _ in range(batch_size)]

def straight(observation):
    """Go straight ahead."""
    batch_size = observation.shape[0]
    # [steer, gas, brake]
    return np.array([[0.0, 1.0, 0.0] for _ in range(batch_size)], dtype=np.float32)

# Random relative to action space...
