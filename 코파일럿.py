import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# 환경 설정
GRID_SIZE = 9
START = (0, 0)
GOAL = (8, 8)
OBSTACLE = (4, 4)

class Environment:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
        self.grid[OBSTACLE] = -1  # 장애물
        self.state = START

    def reset(self):
        self.state = START
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:  # 위
            x = max(0, x - 1)
        elif action == 1:  # 아래
            x = min(GRID_SIZE - 1, x + 1)
        elif action == 2:  # 왼쪽
            y = max(0, y - 1)
        elif action == 3:  # 오른쪽
            y = min(GRID_SIZE - 1, y + 1)

        next_state = (x, y)
        if next_state == GOAL:
            reward = 1
            done = True
        elif next_state == OBSTACLE:
            reward = -1
            done = False
        else:
            reward = 0
            done = False

        self.state = next_state
        return next_state, reward, done

# DQN 모델 설정
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(2, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self):
        self.model = DQN()
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(4)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.target_model(next_state)).item()
            target_f = self.model(state).detach()
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# 학습 과정
env = Environment()
agent = DQNAgent()
episodes = 500

for e in range(episodes):
    state = env.reset()
    state = np.array(state)
    for time in range(200):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        next_state = np.array(next_state)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
    agent.replay(32)
    agent.update_target_model()

print("학습 완료!")
