import numpy as np
import gym
import os
from tqdm import tqdm
from cartpole import Agent  # Assuming cartpole.py contains the Agent class

def test(env):
    """
    Test the agent on the given environment.
    Parameters:
        env: the given environment.
    Returns:
        None (Don't need to return anything)
    """
    testing_agent = Agent(env)

    testing_agent.qtable = np.load("./Tables/cartpole_table.npy")
    rewards = []

    for _ in range(1):
        observation, _ = env.reset()  # Unpack observation from env.reset()
        state = testing_agent.discretize_observation(observation)
        count = 0
        while True:
            count += 1
            action = np.argmax(testing_agent.qtable[tuple(state)])
            next_observation, _, done, _, _ = testing_agent.env.step(action)

            if done:
                rewards.append(count)
                break

            next_state = testing_agent.discretize_observation(next_observation)
            state = next_state

    print(f"average reward: {np.mean(rewards)}")
    print(f"max Q:{testing_agent.check_max_Q()}")




if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode='human')

    # Testing section:
    test(env)
    env.close()