import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import tqdm
from collections import deque, namedtuple
from torch.utils.tensorboard import SummaryWriter
import ale_py

gym.register_envs(ale_py)

# === Model Definition ===
class DQN(nn.Module):
    def __init__(self, k_frames=4, n_actions=7):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(k_frames, 32, kernel_size=8, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        with torch.no_grad():
            dummy = torch.zeros(1, k_frames, 250, 160)
            x = F.relu(self.pool1(self.conv1(dummy)))
            x = F.relu(self.pool2(self.conv2(x)))
            x = F.relu(self.pool3(self.conv3(x)))
            flatten_size = x.view(1, -1).shape[1]

        self.fc = nn.Linear(flatten_size, 512)
        self.out = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.out(x)

# === Replay Memory ===
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity=20000):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size=100):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# === Preprocessing ===
def preprocess(frame):
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (160, 250))
    _, binary = cv2.threshold(resized, 1, 255, cv2.THRESH_BINARY)
    return binary / 255.0

# === Training Setup ===
env = gym.make("ALE/Assault-v5")
n_actions = env.action_space.n
k = 4

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
policy_net = DQN(k, n_actions).to(device)
target_net = DQN(k, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
memory = ReplayMemory()

epsilon = 0.8
epsilon_min = 0.05
epsilon_decay = (epsilon - epsilon_min) / 50000
gamma = 0.99

writer = SummaryWriter("runs/dqn_assault")

# === Action Selection ===
def select_action(state):
    global epsilon
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        return policy_net(state).argmax(dim=1).item()

# === Model Optimization ===
def optimize_model():
    if len(memory) < 100:
        return
    transitions = memory.sample(100)
    batch = Transition(*zip(*transitions))

    state_batch = torch.tensor(np.stack(batch.state), dtype=torch.float32).to(device)
    action_batch = torch.tensor(batch.action).unsqueeze(1).to(device)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)
    next_state_batch = torch.tensor(np.stack(batch.next_state), dtype=torch.float32).to(device)
    done_batch = torch.tensor(batch.done, dtype=torch.float32).to(device)

    q_values = policy_net(state_batch).gather(1, action_batch).squeeze()
    next_q_values = target_net(next_state_batch).max(1)[0].detach()
    expected_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)

    loss = F.mse_loss(q_values, expected_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar("Loss/train", loss.item(), optimize_model.step_count)
    optimize_model.step_count += 1
optimize_model.step_count = 0

# === Training Loop ===
num_episodes = 50000
reward_buffer = deque(maxlen=50)

for episode in tqdm.trange(num_episodes, desc="Training DQN"):
    frames = []
    obs, _ = env.reset()
    for _ in range(k):
        frames.append(preprocess(obs))
    state = np.stack(frames, axis=0)[None, ...]

    total_reward = 0
    done = False
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        action = select_action(state_tensor)
        obs_next, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        next_frame = preprocess(obs_next)
        next_state = np.concatenate([state[0, 1:], [next_frame]], axis=0)[None, ...]

        memory.push(state[0], action, reward, next_state[0], done)
        state = next_state

        optimize_model()

    writer.add_scalar("Reward/episode", total_reward, episode)
    writer.add_scalar("Epsilon", epsilon, episode)
    reward_buffer.append(total_reward)

    if (episode + 1) % 50 == 0:
        avg_reward = sum(reward_buffer) / len(reward_buffer)
        writer.add_scalar("Reward/avg_50_episodes", avg_reward, episode)
        tqdm.tqdm.write(f"Episode {episode}, Avg Reward (last 50): {avg_reward:.2f}")

    if episode % 1000 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(epsilon_min, epsilon - epsilon_decay)

writer.close()