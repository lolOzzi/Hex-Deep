import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import gym

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Q-Network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
lr = 1e-3
batch_size = 64
replay_buffer = []
max_buffer_size = 10000

# Training loop
def train_dqn(env_name="CartPole-v1", episodes=500):
    # Specify the render mode when creating the environment
    env = gym.make(env_name, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_network = DQN(state_dim, action_dim).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    global epsilon
    for episode in range(episodes):
        state = torch.tensor(env.reset()[0], dtype=torch.float32).to(device)
        total_reward = 0

        done = False
        while not done:
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q_values = q_network(state)
                    action = torch.argmax(q_values).item()

            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > max_buffer_size:
                replay_buffer.pop(0)

            state = next_state
            total_reward += reward

            # Train
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.stack(states)
                next_states = torch.stack(next_states)
                actions = torch.tensor(actions, device=device)
                rewards = torch.tensor(rewards, device=device)
                dones = torch.tensor(dones, dtype=torch.float32, device=device)

                q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
                with torch.no_grad():
                    max_next_q = q_network(next_states).max(1)[0]
                    target_q = rewards + gamma * max_next_q * (1 - dones)

                loss = loss_fn(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Render every 100 episodes
        if (episode + 1) % 100 == 0:
            print(f"Rendering Episode {episode + 1}...")
            render_episode(env, q_network)

        print(f"Episode {episode + 1}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    env.close()
    print("Training complete.")

# Render a single episode
def render_episode(env, q_network):
    state = torch.tensor(env.reset()[0], dtype=torch.float32).to(device)
    done = False
    while not done:
        action = torch.argmax(q_network(state)).item()
        next_state, reward, done, _, _ = env.step(action)
        state = torch.tensor(next_state, dtype=torch.float32).to(device)
        env.render()  # Will render the environment window
    env.close()

# Run training
if __name__ == "__main__":
    train_dqn()
