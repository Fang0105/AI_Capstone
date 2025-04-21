import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import cv2
from collections import deque, namedtuple
import gymnasium as gym
import ale_py
from torch.utils.tensorboard import SummaryWriter
import tqdm
import matplotlib.pyplot as plt

# --- Register env
gym.register_envs(ale_py)
env = gym.make("ALE/Assault-v5", render_mode="rgb_array")
n_actions = env.action_space.n
k = 4  # stacked frames
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# --- TensorBoard Writer
writer = SummaryWriter(log_dir="runs/AssaultDQN_10000_clipped")

# --- Preprocessing
def preprocess(frame):
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(frame, (84, 84))
    return resized / 255.0

# --- Replay Memory (increased capacity)
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity=10000):  # Increased capacity to 50000
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size=100):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# --- DQN Model (Enhanced Capacity)
class DQN(nn.Module):
    def __init__(self, k_frames=4, n_actions=7):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(k_frames, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(7 * 7 * 64, 2048)  # Increased from 512 to 2048
        self.out = nn.Linear(2048, n_actions)  # Increased from 512 to 2048

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.out(x)

# --- Initialize networks
policy_net = DQN(k, n_actions).to(device)
target_net = DQN(k, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=0.00025)
memory = ReplayMemory(capacity=50000)  # Increased capacity to 50000

# --- ε-greedy settings (Exponential Decay)
epsilon_start = 1.0
epsilon_min = 0.2  # Increased minimum epsilon to prolong exploration phase
decay_rate = 100000  # Slower decay for epsilon
steps_done = 0

gamma = 0.97
target_update_steps = 1000
batch_size = 32
warmup_steps = 10000

# --- Action selection with exponential decay
def select_action(state):
    global steps_done
    epsilon = epsilon_min + (epsilon_start - epsilon_min) * np.exp(-steps_done / decay_rate)
    steps_done += 1
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        return policy_net(state).argmax(dim=1).item()

# --- Optimize model
loss_list = []
def optimize_model():
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.tensor(np.stack(batch.state), dtype=torch.float32).to(device)
    action_batch = torch.tensor(batch.action).unsqueeze(1).to(device)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)
    next_state_batch = torch.tensor(np.stack(batch.next_state), dtype=torch.float32).to(device)
    done_batch = torch.tensor(batch.done, dtype=torch.float32).to(device)

    q_values = policy_net(state_batch).gather(1, action_batch).squeeze()
    next_actions = policy_net(next_state_batch).argmax(1, keepdim=True)
    next_q_values = target_net(next_state_batch).gather(1, next_actions).squeeze().detach()
    expected_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)

    loss = F.smooth_l1_loss(q_values, expected_q_values)
    writer.add_scalar("Loss/train", loss.item(), steps_done)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- Training loop
num_episodes = 10000
update_counter = 0
reward_history = []

for episode in tqdm.trange(num_episodes):
    obs, _ = env.reset()
    obs = obs[40:190, 0:160]  # Crop the observation
    frames = [preprocess(obs)] * k
    state = np.stack(frames, axis=0)[None, ...]

    total_reward = 0
    true_total_reward = 0
    done = False

    while not done:
        state_tensor = torch.from_numpy(state).float().to(device)
        action = select_action(state_tensor)
        obs_next, reward, terminated, truncated, _ = env.step(action)
        obs_next = obs_next[40:190, 0:160] # Crop the observation
        done = terminated or truncated

        true_total_reward += reward
        reward = np.clip(reward, -1.0, 1.0)
        total_reward += reward
        next_frame = preprocess(obs_next)
        next_state = np.concatenate([state[0, 1:], [next_frame]], axis=0)[None, ...]

        memory.push(state[0], action, reward, next_state[0], done)
        state = next_state

        if steps_done > warmup_steps:
            optimize_model()
            update_counter += 1
            if update_counter % target_update_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())

    reward_history.append(true_total_reward)
    epsilon = epsilon_min + (epsilon_start - epsilon_min) * np.exp(-steps_done / decay_rate)
    writer.add_scalar("Reward/true_episode", true_total_reward, episode)
    writer.add_scalar("Reward/clipped_episode", total_reward, episode)
    writer.add_scalar("Epsilon", epsilon, episode)

    if episode % 50 == 0:
        avg_reward = np.mean(reward_history[-50:])
        writer.add_scalar("Reward/avg_50_true", avg_reward, episode)
        torch.save(policy_net.state_dict(), f"training_result_clipped/dqn_episode_{episode}.pt")
        print(f"Episode {episode}, Avg True Reward (last 50): {avg_reward:.2f}")

# --- Save final reward plot
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("True Total Reward")
plt.title("Reward Trend (True)")
plt.grid(True)
plt.savefig("training_result_clipped/reward_trend_true.png")

writer.close()
