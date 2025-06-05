import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

BOARD_SIZE = 3
EPSILON = 0.2
GAMMA = 0.99
LR = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 64
TARGET_UPDATE = 10
EPISODES = 20000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Game logic
class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.done = False
        self.winner = None
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        return self.board.flatten().astype(np.float32)

    def available_actions(self):
        return [i for i in range(9) if self.board[i // 3][i % 3] == 0]

    def step(self, action):
        if self.done or self.board[action // 3][action % 3] != 0:
            return self.get_state(), -10, True  # Illegal move

        self.board[action // 3][action % 3] = self.current_player
        reward, self.done, self.winner = self.check_winner()

        state = self.get_state()
        self.current_player *= -1
        return state, reward, self.done

    def check_winner(self):
        for i in range(3):
            if abs(sum(self.board[i])) == 3:
                return 1, True, np.sign(sum(self.board[i]))
            if abs(sum(self.board[:, i])) == 3:
                return 1, True, np.sign(sum(self.board[:, i]))

        diag1 = sum(self.board[i][i] for i in range(3))
        diag2 = sum(self.board[i][2 - i] for i in range(3))
        if abs(diag1) == 3:
            return 1, True, np.sign(diag1)
        if abs(diag2) == 3:
            return 1, True, np.sign(diag2)

        if not any(0 in row for row in self.board):
            return 0.5, True, 0  # Draw
        return 0, False, None


# DQN
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )

    def forward(self, x):
        return self.fc(x)


# Replay memory
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity

    def push(self, transition):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Agent
class Agent:
    def __init__(self):
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.steps_done = 0
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def select_action(self, state, available_actions, epsilon=EPSILON):
        if random.random() < epsilon:
            return random.choice(available_actions)
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor)[0].cpu().numpy()
            q_values = [q if i in available_actions else -np.inf for i, q in enumerate(q_values)]
            return int(np.argmax(q_values))

    def optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = self.memory.sample(BATCH_SIZE)
        state_batch = torch.tensor([t[0] for t in batch], dtype=torch.float32).to(device)
        action_batch = torch.tensor([t[1] for t in batch]).unsqueeze(1).to(device)
        reward_batch = torch.tensor([t[2] for t in batch], dtype=torch.float32).to(device)
        next_state_batch = torch.tensor([t[3] for t in batch], dtype=torch.float32).to(device)
        done_batch = torch.tensor([t[4] for t in batch], dtype=torch.float32).to(device)

        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
        expected_q = reward_batch + GAMMA * next_q_values * (1 - done_batch)

        loss = nn.MSELoss()(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
def render(board):
    symbols = {1: "X", -1: "O", 0: " "}
    for row in board.reshape(3, 3):
        print(" | ".join(symbols[int(cell)] for cell in row))
        print("-" * 9)
    print("")


# Self-play training loop
agent = Agent()

for episode in range(EPISODES):
    env = TicTacToe()
    state = env.reset()
    total_reward = 0

    while not env.done:
        available = env.available_actions()
        action = agent.select_action(state, available)
        next_state, reward, done = env.step(action)

        # Opponent (self)
        if not done:
            opp_available = env.available_actions()
            opp_action = agent.select_action(next_state, opp_available)
            next_state, reward, done = env.step(opp_action)
            reward = -reward  # Agent sees opponent's reward as negative

        agent.memory.push((state, action, reward, next_state, float(done)))
        state = next_state
        total_reward += reward
        agent.optimize()

    if episode % TARGET_UPDATE == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())

    if episode % 100 == 0:
        print(f"\nEpisode {episode}, Total reward: {total_reward}")
        # Visualize a self-play game
        test_env = TicTacToe()
        test_state = test_env.reset()
        print("Self-play visualization:")
        render(test_env.board)

        while not test_env.done:
            avail = test_env.available_actions()
            act = agent.select_action(test_state, avail, epsilon=0.0)
            test_state, _, _ = test_env.step(act)
            render(test_env.board)

            if not test_env.done:
                opp_avail = test_env.available_actions()
                opp_act = agent.select_action(test_state, opp_avail, epsilon=0.0)
                test_state, _, _ = test_env.step(opp_act)
                render(test_env.board)

        if test_env.winner == 1:
            print("X wins")
        elif test_env.winner == -1:
            print("O wins")
        else:
            print("Draw")


