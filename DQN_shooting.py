import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import pygame
from gym import Env
from gym.spaces import Discrete, Box

# ShootingGameEnv 환경 정의
class ShootingGameEnv(Env):
    def __init__(self):
        super(ShootingGameEnv, self).__init__()
        pygame.init()
        self.screen_width = 200
        self.screen_height = 300
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("RL Shooting Game")

        self.player_width = 25
        self.player_height = 5
        self.enemy_width = 15
        self.enemy_height = 15

        self.action_space = Discrete(3)  # 0: 왼쪽, 1: 정지, 2: 오른쪽
        self.observation_space = Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 1), dtype=np.uint8)

        self.reset()

    def reset(self):
        self.player_x = self.screen_width // 2 - self.player_width // 2
        self.player_y = self.screen_height - self.player_height - 10
        self.enemy_x = np.random.randint(0, self.screen_width - self.enemy_width)
        self.enemy_y = 0
        self.score = 0
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        state_surface = pygame.Surface((self.screen_width, self.screen_height))
        state_surface.fill((0, 0, 0))
        pygame.draw.rect(state_surface, (0, 255, 0), (self.player_x, self.player_y, self.player_width, self.player_height))
        pygame.draw.rect(state_surface, (255, 0, 0), (self.enemy_x, self.enemy_y, self.enemy_width, self.enemy_height))
        state = pygame.surfarray.array3d(state_surface)
        state = np.transpose(state, (1, 0, 2))
        state = np.mean(state, axis=2, keepdims=True).astype(np.uint8)
        return state

    def step(self, action):
        if action == 0:
            self.player_x = max(0, self.player_x - 5)
        elif action == 2:
            self.player_x = min(self.screen_width - self.player_width, self.player_x + 5)

        self.enemy_y += 3

        reward = 0
        if self.enemy_y + self.enemy_height >= self.player_y:
            if self.player_x <= self.enemy_x <= self.player_x + self.player_width or \
               self.player_x <= self.enemy_x + self.enemy_width <= self.player_x + self.player_width:
                reward = 1
                self.enemy_x = np.random.randint(0, self.screen_width - self.enemy_width)
                self.enemy_y = 0
                self.score += 1
            else:
                self.done = True

        if self.enemy_y > self.screen_height:
            self.enemy_x = np.random.randint(0, self.screen_width - self.enemy_width)
            self.enemy_y = 0

        return self._get_obs(), reward, self.done, {}

    def render(self, mode='human'):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (0, 255, 0), (self.player_x, self.player_y, self.player_width, self.player_height))
        pygame.draw.rect(self.screen, (255, 0, 0), (self.enemy_x, self.enemy_y, self.enemy_width, self.enemy_height))
        pygame.display.flip()

    def close(self):
        pygame.quit()


# DQN 신경망 정의
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# 하이퍼파라미터 설정 
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32
memory_size = 10000
num_episodes = 10

# 환경 및 네트워크 초기화
env = ShootingGameEnv()
q_network = DQN(env.observation_space.shape[0] * env.observation_space.shape[1], env.action_space.n)
target_network = DQN(env.observation_space.shape[0] * env.observation_space.shape[1], env.action_space.n)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)

# 행동 선택 함수
def select_action(state, epsilon):
    if random.random() < epsilon:
        action = env.action_space.sample()  # Random action (exploration)
        exploration = True
    else:
        state_tensor = torch.FloatTensor(state).flatten().unsqueeze(0)
        with torch.no_grad():
            q_values = q_network(state_tensor)
        action = q_values.argmax().item()  # Greedy action (exploitation)
        exploration = False
    return action, exploration

# 학습 루프
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for t in range(200):
        env.render()  # 그래픽 출력
        action, exploration = select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)

        # 상태, 행동, 보상 출력
        print(f"Episode {episode}, Step {t}")
        print(f"State shape: {state.shape}")
        print(f"Action: {action} {'(Exploration)' if exploration else '(Exploitation)'}")
        print(f"Reward: {reward}")
        print("------")

        memory.append((state, action, reward, next_state, done))
        total_reward += reward
        state = next_state

        if done:
            print(f"Episode {episode} ended after {t + 1} steps")
            break

        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states)).view(batch_size, -1)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(np.array(next_states)).view(batch_size, -1)
            dones = torch.FloatTensor(dones).unsqueeze(1)

            with torch.no_grad():
                next_q_values = target_network(next_states).max(1, keepdim=True)[0]
                td_target = rewards + gamma * next_q_values * (1 - dones)

            current_q_values = q_network(states).gather(1, actions)
            loss = nn.MSELoss()(current_q_values, td_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode} finished with total reward: {total_reward}, epsilon: {epsilon:.4f}")

env.close()
