import cv2
import numpy as np
import torch
import gymnasium as gym
import ale_py
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F

k = 4

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


clipped = False  # Set to True if you want to use the clipped version of the model


# --- Register environment and set up the device
gym.register_envs(ale_py)
env = gym.make("ALE/Assault-v5", render_mode="rgb_array")
n_actions = env.action_space.n
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# --- Load trained model
policy_net = DQN(k, n_actions).to(device)
policy_net.load_state_dict(torch.load("training_result/dqn_episode_4500.pt"))  # Load your trained model
policy_net.eval()

# --- Preprocessing (same as in training)
def preprocess(frame):
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(frame, (84, 84))
    return resized / 255.0

# --- Action selection function (same as in training)
def select_action(state):
    with torch.no_grad():
        return policy_net(state).argmax(dim=1).item()

# --- Testing loop and video recording
num_test_episodes = 10  # You can adjust the number of test episodes
# video_writer = cv2.VideoWriter("test_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (84, 84))  # Adjust the frame size as needed

all_rewards = []

for episode in range(num_test_episodes):
    obs, _ = env.reset()
    if clipped:
        obs = obs[40:190, 0:160]
    frames = [preprocess(obs)] * k
    state = np.stack(frames, axis=0)[None, ...]

    total_reward = 0
    done = False

    while not done:
        state_tensor = torch.from_numpy(state).float().to(device)
        action = select_action(state_tensor)
        obs_next, reward, terminated, truncated, _ = env.step(action)
        if clipped:
            obs_next = obs_next[40:190, 0:160]
        done = terminated or truncated

        # Capture frame and write to video
        frame = preprocess(obs_next) * 255  # Scale back to 0-255 for display
        frame = frame.astype(np.uint8)  # Convert to uint8 type

        total_reward += reward
        next_state = np.concatenate([state[0, 1:], [frame]], axis=0)[None, ...]

        state = next_state

    print(f"Episode {episode}, Total Reward: {total_reward}")
    all_rewards.append(total_reward)

print(f"Average Reward over {num_test_episodes} episodes: {np.mean(all_rewards)}")
print(f"Maximum Reward over {num_test_episodes} episodes: {np.max(all_rewards)}")
print(f"Minimum Reward over {num_test_episodes} episodes: {np.min(all_rewards)}")
env.close()

