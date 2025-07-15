import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# GPU 사용 가능 여부 확인 및 device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

'''class GridEnvironment:
    def __init__(self):
        self.grid_size = 10
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.start = (0, 0)
        self.small_goal = (3, 3)  # 목표 위치는 여기서 변경
        self.large_goal = (4, 3)
        self.reset()
        
    def reset(self):
        self.position = self.start
        return self._get_state()
        
    def _get_state(self):
        x, y = self.position
        small_x, small_y = self.small_goal
        large_x, large_y = self.large_goal
        return np.array([
            x / (self.grid_size - 1),
            y / (self.grid_size - 1),
            small_x / (self.grid_size - 1),
            small_y / (self.grid_size - 1),
            large_x / (self.grid_size - 1),
            large_y / (self.grid_size - 1)
        ])

    def step(self, action):
        x, y = self.position
        if action == 0 and x > 0:  # 상
            x -= 1
        elif action == 1 and x < self.grid_size - 1:  # 하
            x += 1
        elif action == 2 and y > 0:  # 좌
            y -= 1
        elif action == 3 and y < self.grid_size - 1:  # 우
            y += 1

        self.position = (x, y)
        
        reward = -0.1  # 기본적인 스텝 페널티
        
        if self.position == self.small_goal:
            return self._get_state(), 1, True
        elif self.position == self.large_goal:
            return self._get_state(), 10, True
        else:
            dist_to_small = abs(x - self.small_goal[0]) + abs(y - self.small_goal[1])
            dist_to_large = abs(x - self.large_goal[0]) + abs(y - self.large_goal[1])
            reward += 0.1 * (1.0 / (dist_to_small + 1) + 1.0 / (dist_to_large + 1))
            
            return self._get_state(), reward, False'''

import numpy as np

class GridEnvironment:
    def __init__(self):
        self.grid_size = 10
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.start = (0, 0)
        self.small_goal_range = ((3, 3), (7, 7))  # 작은 목표 범위 (좌상단, 우하단)
        #self.large_goal_range = ((5, 2), (7, 4))  # 큰 목표 범위 (좌상단, 우하단)
        self.reset()
        
    def reset(self):
        self.position = self.start
        return self._get_state()
        
    def _get_state(self):
        x, y = self.position
        # 범위의 중심을 상태에 포함
        small_x_center = (self.small_goal_range[0][0] + self.small_goal_range[1][0]) / 2
        small_y_center = (self.small_goal_range[0][1] + self.small_goal_range[1][1]) / 2
        '''large_x_center = (self.large_goal_range[0][0] + self.large_goal_range[1][0]) / 2
        large_y_center = (self.large_goal_range[0][1] + self.large_goal_range[1][1]) / 2'''
        return np.array([
            x / (self.grid_size - 1),
            y / (self.grid_size - 1),
            small_x_center / (self.grid_size - 1),
            small_y_center / (self.grid_size - 1),
            '''large_x_center / (self.grid_size - 1),
            large_y_center / (self.grid_size - 1)'''
        ])

    def step(self, action):
        x, y = self.position
        if action == 0 and x > 0:  # 상
            x -= 1
        elif action == 1 and x < self.grid_size - 1:  # 하
            x += 1
        elif action == 2 and y > 0:  # 좌
            y -= 1
        elif action == 3 and y < self.grid_size - 1:  # 우
            y += 1

        self.position = (x, y)
        
        reward = +0.1  # 기본적인 스텝 페널티
        
        # 범위 안에 있는지 확인
        if self._is_in_range(self.position, self.small_goal_range):
            return self._get_state(), -10, True
        
        else:
            # 거리 기반 보상 계산
            dist_to_small = self._min_dist_to_range(self.position, self.small_goal_range)
            #dist_to_large = self._min_dist_to_range(self.position, self.large_goal_range)
            reward += 0.1 
            
            return self._get_state(), reward, False 
    
    def _is_in_range(self, position, goal_range):
        """포지션이 범위(goal_range) 안에 있는지 확인"""
        x, y = position
        (x1, y1), (x2, y2) = goal_range
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def _min_dist_to_range(self, position, goal_range):
        """포지션에서 목표 범위까지의 최소 맨해튼 거리 계산"""
        x, y = position
        (x1, y1), (x2, y2) = goal_range
        
        # x 방향 거리 계산
        if x < x1:
            dx = x1 - x
        elif x > x2:
            dx = x - x2
        else:
            dx = 0
        
        # y 방향 거리 계산
        if y < y1:
            dy = y1 - y
        elif y > y2:
            dy = y - y2
        else:
            dy = 0
        
        return dx + dy


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

def train_dqn(env):
    state_dim = 6
    action_dim = 4
    
    model = DQN(state_dim, action_dim).to(device)
    target_model = DQN(state_dim, action_dim).to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    replay_buffer = []
    max_buffer_size = 10000
    batch_size = 128
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.997
    epsilon_min = 0.01
    episodes = 1000
    target_update = 5

    rewards_history = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        max_steps = 100

        while not done and steps < max_steps:
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1

            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > max_buffer_size:
                replay_buffer.pop(0)

            state = next_state

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(np.array(states)).to(device)
                next_states = torch.FloatTensor(np.array(next_states)).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                dones = torch.FloatTensor(dones).to(device)

                current_q_values = model(states).gather(1, actions).squeeze()

                with torch.no_grad():
                    next_q_values = target_model(next_states).max(1)[0]
                    target_q_values = rewards + (1 - dones) * gamma * next_q_values

                loss = criterion(current_q_values, target_q_values)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        
        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())

        rewards_history.append(total_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = sum(rewards_history[-10:]) / 10
            print(f"Episode {episode + 1}: Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")

    return model

def visualize_path_with_animation(env, model):
    state = np.array(env.reset())
    done = False
    path = [env.start]

    fig, ax = plt.subplots(figsize=(8, 8))
    grid = np.zeros((env.grid_size, env.grid_size))

    def render():
        grid[:, :] = 0
        grid[env.start] = 0.3
        grid[env.small_goal] = 0.6
        grid[env.large_goal] = 1.0
        for (x, y) in path:
            grid[x, y] = 0.8
        
        current_pos = path[-1]
        
        ax.clear()
        ax.imshow(grid, cmap="Blues")
        
        for i in range(env.grid_size):
            for j in range(env.grid_size):
                ax.text(j, i, f'({i},{j})', ha='center', va='center', color='red', fontsize=8)
                ax.axhline(y=i-0.5, color='black', linewidth=0.5)
                ax.axvline(x=j-0.5, color='black', linewidth=0.5)
        
        ax.text(-0.5, -0.5, "S: Start", color='black')
        ax.text(1.5, -0.5, "G1: Small Goal", color='black')
        ax.text(3.5, -0.5, "G2: Large Goal", color='black')
        
        ax.plot(current_pos[1], current_pos[0], 'r*', markersize=15)
        
        ax.set_xticks(range(env.grid_size))
        ax.set_yticks(range(env.grid_size))
        ax.grid(True)
        ax.set_title(f'Current Position: {current_pos}')
        plt.draw()
        plt.pause(0.5)

    while not done:
        render()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
        action = q_values.argmax().item()
        next_state, _, done = env.step(action)
        path.append(env.position)
        state = np.array(next_state)

    render()
    plt.show()

# 실행
env = GridEnvironment()
trained_model = train_dqn(env)
visualize_path_with_animation(env, trained_model)