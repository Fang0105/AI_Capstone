import torch
import gym
import numpy as np
from DQN import Agent  # Assuming DQN.py contains the Agent class

def test(env):
    """
    Test the agent on the given environment.
    Parameters:
        env: the given environment.
    Returns:
        None
    """
    rewards = []
    testing_agent = Agent(env)
    testing_agent.target_net.load_state_dict(torch.load("./Tables/DQN.pt"))
    for _ in range(1):
        state = env.reset()[0]  # Updated for correct state format
        count = 0
        while True:
            count += 1
            Q = testing_agent.target_net.forward(torch.FloatTensor(state)).squeeze(0).detach()
            action = int(torch.argmax(Q).numpy())
            
            # Updated to handle the new return signature
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # Define done as the combination of terminated and truncated
            
            if done:
                rewards.append(count)
                break
            state = next_state

    print(f"reward: {np.mean(rewards)}")
    print(f"max Q:{testing_agent.check_max_Q()}")

if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode='human')        
        
    # Testing section:
    test(env)
    env.close()

