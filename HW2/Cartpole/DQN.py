import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import random
from collections import deque
import os
from tqdm import tqdm

total_rewards = []


class replay_buffer():
    '''
    A deque storing trajectories
    '''
    def __init__(self, capacity):
        self.capacity = capacity  # the size of the replay buffer
        self.memory = deque(maxlen=capacity)  # replay buffer itself

    def insert(self, state, action, reward, next_state, done):
        '''
        Insert a sequence of data gotten by the agent into the replay buffer.
        Parameter:
            state: the current state
            action: the action done by the agent
            reward: the reward agent got
            next_state: the next state
            done: the status showing whether the episode finish        
        Return:
            None
        '''
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        '''
        Sample a batch size of data from the replay buffer.
        Parameter:
            batch_size: the number of samples which will be propagated through the neural network
        Returns:
            observations: a batch size of states stored in the replay buffer
            actions: a batch size of actions stored in the replay buffer
            rewards: a batch size of rewards stored in the replay buffer
            next_observations: a batch size of "next_state"s stored in the replay buffer
            done: a batch size of done stored in the replay buffer
        '''
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, done = zip(*batch)
        return observations, actions, rewards, next_observations, done

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    '''
    The structure of the Neural Network calculating Q values of each state.
    '''
    def __init__(self,  num_actions, hidden_layer_size=50):
        super(Net, self).__init__()
        self.input_state = 4  # the dimension of state space
        self.num_actions = num_actions  # the dimension of action space
        self.fc1 = nn.Linear(self.input_state, 32)  # input layer
        self.fc2 = nn.Linear(32, hidden_layer_size)  # hidden layer
        self.fc3 = nn.Linear(hidden_layer_size, num_actions)  # output layer

    def forward(self, states):
        '''
        Forward the state to the neural network.        
        Parameter:
            states: a batch size of states
        Return:
            q_values: a batch size of q_values
        '''
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class Agent():
    def __init__(self, env, epsilon=0.95, learning_rate=0.0002, GAMMA=0.97, batch_size=32, capacity=10000):
        """
        The agent learning how to control the action of the cart pole.
        Hyperparameters:
            epsilon: Determines the explore/exploit rate of the agent
            learning_rate: Determines the step size while moving toward a minimum of a loss function
            GAMMA: the discount factor (tradeoff between immediate rewards and future rewards)
            batch_size: the number of samples which will be propagated through the neural network
            capacity: the size of the replay buffer/memory
        """
        self.env = env
        self.n_actions = 2  # the number of actions
        self.count = 0

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.batch_size = batch_size
        self.capacity = capacity

        self.buffer = replay_buffer(self.capacity)
        self.evaluate_net = Net(self.n_actions)  # the evaluate network
        self.target_net = Net(self.n_actions)  # the target network

        self.optimizer = torch.optim.Adam(
            self.evaluate_net.parameters(), lr=self.learning_rate)  # Adam is a method used to optimize the neural network

    def learn(self):
        '''
        - Implement the learning function.
        - Here are the hints to implement.
        Steps:
        -----
        1. Update target net by current net every 100 times. (we have done this for you)
        2. Sample trajectories of batch size from the replay buffer.
        3. Forward the data to the evaluate net and the target net.
        4. Compute the loss with MSE.
        5. Zero-out the gradients.
        6. Backpropagation.
        7. Optimize the loss function.
        -----
        Parameters:
            self: the agent itself.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        Returns:
            None
        '''
        if self.count % 100 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())

        # randomly choose batch_size samples from the buffer(memory)
        states, actions, rewards, next_states, done = self.buffer.sample(self.batch_size) 

        # convert features to tensor
        states_tensor = torch.FloatTensor(np.array(states))
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)    
        next_states_tensor = torch.FloatTensor(np.array(next_states))
        done_tensor = torch.BoolTensor(done)

        # use evaluate_net to calculate the q-value of each state in states
        q_eval = self.evaluate_net(states_tensor).gather(1, actions_tensor.reshape(self.batch_size, 1))

        # use target_net to calculate the q-value of each next_state in next_states
        q_next = self.target_net(next_states_tensor).detach()

        # calculate the max q-value, R + γ*max(Q(s_next, a_next))*(!done)
        q_target = rewards_tensor.reshape(self.batch_size, 1) + self.gamma * q_next.max(1)[0].reshape(self.batch_size, 1) * (~done_tensor).reshape(self.batch_size, 1)
        
        # calculate the mseloss of states' q-value and next_states' q-value
        loss_func = nn.MSELoss()
        loss = loss_func(q_eval, q_target)

        self.optimizer.zero_grad() # zero-out the gradients
        loss.backward() # backpropagation
        self.optimizer.step() # Optimize the loss function

        torch.save(self.target_net.state_dict(), "./Tables/DQN.pt")

    def choose_action(self, state):
        """
        - Implement the action-choosing function.
        - Choose the best action with given state and epsilon
        Parameters:
            self: the agent itself.
            state: the current state of the environment.
        Returns:
            action: the chosen action.
        """
        with torch.no_grad():
            # Convert state to a tensor, and use unsqueeze to add 1 dimension to satisfy the shape of input
            x = torch.unsqueeze(torch.FloatTensor(state), 0)

            tradeoff = np.random.uniform(0, 1)  # determine if the next action is exploit or explore

            if tradeoff < self.epsilon:
                # Exploit, choose the action with the biggest q-value
                actions_value = self.evaluate_net(x)  # get the q-value of all actions
                action = torch.max(actions_value, 1)[1].data.numpy()[0]  # choose the action with the biggest q-value
            else:
                # Explore, randomly choose an action from action_space
                action = np.random.randint(0, self.n_actions)

        return action

    def check_max_Q(self):
        """
        - Implement the function calculating the max Q value of initial state(self.env.reset()).
        - Check the max Q value of initial state        
        Parameter:
            self: the agent itself.
        Return:
            max_q: the max Q value of initial state(self.env.reset())
        """
        max_q = torch.max(self.target_net(torch.FloatTensor(self.env.reset()[0])))
        return max_q


def train(env):
    """
    Train the agent on the given environment.
    Parameters:
        env: the given environment.
    Returns:
        None
    """
    agent = Agent(env)
    episode = 1000
    rewards = []
    for _ in tqdm(range(episode)):
        state = env.reset()[0]  # Updated for correct state format
        count = 0
        while True:
            count += 1
            agent.count += 1
            action = agent.choose_action(state)
            
            # Updated to handle the new return signature
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # Define done as the combination of terminated and truncated
            
            agent.buffer.insert(state, int(action), reward, next_state, int(done))
            
            if len(agent.buffer) >= 1000:
                agent.learn()
            if done:
                rewards.append(count)
                break
            state = next_state
    total_rewards.append(rewards)


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
    for _ in range(100):
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
    env = gym.make('CartPole-v1')        
    os.makedirs("./Tables", exist_ok=True)

    # Training section:
    for i in range(5):
        print(f"#{i + 1} training progress")
        train(env)
        
    # Testing section:
    test(env)
    env.close()

    os.makedirs("./Rewards", exist_ok=True)
    np.save("./Rewards/DQN_rewards.npy", np.array(total_rewards))
