import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



class GridEnvironment:
    def __init__(self):
        
        self.grid = np.zeros((5 ,20))
        self.start = (1, 1)  # 시작 지점
        self.danger_ellipse = {'center': (5, 5), 'a': 2, 'b': 1}  # 위험 영역 (타원)
        self.max_steps = 200
        self.current_steps = 0
        self.direction = 0
        self.position = self.start
        self.last_angle = 0
        self.total_reward = 0  # 누적 보상 추적을 위해 추가
        self.flag = False
        self.prev_dist = 0
        self.reset()

    def reset(self):
        self.position = self.start
        self.direction = 0
        self.current_steps = 0
        self.last_angle = np.arctan2(0, 0)
        self.total_reward = 0  # 리셋 시 누적 보상도 초기화
        self.prev_dist = 0
        self.flag = False
        return self._get_state()

    def _is_in_ellipse(self, position):
        x, y = position
        h, k = self.danger_ellipse['center']
        a, b = self.danger_ellipse['a'], self.danger_ellipse['b']
        return ((x - h) / a) ** 2 + ((y - k) / b) ** 2 <= 1

    def _distance_to_boundary(self, position, angle):
        x, y = position
        dx, dy = np.cos(angle), np.sin(angle)

        t_x = (0 - x) / dx if dx < 0 else (self.grid_size - 1 - x) / dx if dx > 0 else float('inf')
        t_y = (0 - y) / dy if dy < 0 else (self.grid_size - 1 - y) / dy if dy > 0 else float('inf')

        distance_to_boundary = min(max(t_x, 0), max(t_y, 0))
        return distance_to_boundary

    def _distance_to_ellipse(self, position, angle):
        x, y = position
        h, k = self.danger_ellipse['center']
        a, b = self.danger_ellipse['a'], self.danger_ellipse['b']

        # 타원 방정식으로 가장 가까운 점의 거리를 근사적으로 계산
        dx, dy = np.cos(angle), np.sin(angle)
        t = np.linspace(0, 10, 1000)  # 0부터 10까지 샘플링
        points = np.array([x + t * dx, y + t * dy]).T
        distances = ((points[:, 0] - h) / a) ** 2 + ((points[:, 1] - k) / b) ** 2

        within_ellipse = np.where(distances <= 1)[0]
        if within_ellipse.size > 0:
            return t[within_ellipse[0]]
        return float('inf')

    def _distance_in_direction(self, angle):
        radians = np.radians(angle)
        dist_to_boundary = self._distance_to_boundary(self.position, radians)
        dist_to_danger = self._distance_to_ellipse(self.position, radians)
        return min(dist_to_boundary, dist_to_danger)

    def _get_state(self):
        directions = [-90, -45, 0, 45, 90]
        state_info = [
            self._distance_in_direction(self.direction + angle) / self.grid_size
            for angle in directions
        ]

        dx = self.position[0] - self.start[0]
        dy = self.position[1] - self.start[1]
        dist_to_start = np.sqrt(dx**2 + dy**2) / self.grid_size
        current_angle = np.arctan2(dy, dx) / np.pi
        step_info = self.current_steps / self.max_steps

        return np.array(state_info + [dist_to_start, current_angle, step_info])

    def step(self, action):
        rotations = [-90, -45, 0, 45, 90]
        self.direction = (self.direction + rotations[action]) % 360
        radians = np.radians(self.direction)
        dx, dy = 0.5 * np.cos(radians), 0.5 * np.sin(radians)

        new_x = np.clip(self.position[0] + dx, 0, self.grid_size - 1)
        new_y = np.clip(self.position[1] + dy, 0, self.grid_size - 1)
        self.position = (new_x, new_y)
        self.current_steps += 1

        dx = self.position[0] - self.start[0]
        dy = self.position[1] - self.start[1]
        current_angle = np.arctan2(dy, dx)

        dx = self.position[0] - self.start[0]
        dy = self.position[1] - self.start[1]
        dist_to_start = np.sqrt(dx**2 + dy**2)

        reward = 0.1

        # 타원 영역에 들어갔을 때의 처리
        if self._is_in_ellipse(self.position):
            self.total_reward -= 50.0
            return self._get_state(), -50.0, True

        min_distance_to_danger = min([self._distance_in_direction(self.direction + angle) for angle in [-90, -45, 0, 45, 90]])
        if min_distance_to_danger < 0.01:
            return self._get_state(), -50.0, True

        if dist_to_start > 10.0 and not self.flag:
            self.flag = True

        if self.prev_dist > dist_to_start and not self.flag:
            reward -= 8

        if self.prev_dist < dist_to_start and not self.flag: # 시작부터 거리가 아닌 이전 위치와의 거리차이로 해야할듯듯
            reward += 0.5 * dist_to_start

        if self.prev_dist < dist_to_start and self.flag:
            reward -= 8

        if self.prev_dist > dist_to_start and self.flag: # 시작부터 거리가 아닌 이전 위치와의 거리차이로 해야할듯듯
            reward += 0.5 * dist_to_start

        self.prev_dist = dist_to_start

        angle_diff = (current_angle - self.last_angle) % (2 * np.pi)
        if angle_diff > 0:
            reward += 0.2

        self.last_angle = current_angle
        done = self.current_steps >= self.max_steps

        # 누적 보상 업데이트 및 체크
        self.total_reward += reward
        if self.total_reward <= -50:
            return self._get_state(), reward, True

        return self._get_state(), reward, done

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
    state_dim = 8  # 5개 방향 + 시작점 거리 + 각도 + 스텝
    action_dim = 5
    
    model = DQN(state_dim, action_dim).to(device)
    target_model = DQN(state_dim, action_dim).to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.MSELoss()

    replay_buffer = []
    max_buffer_size = 20000
    batch_size = 256
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.01
    episodes = 2000
    target_update = 10

    rewards_history = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, done = env.step(action)
            total_reward += reward

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

    return model, rewards_history




def visualize_episode_steps(env, model, episode_num, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    state = env.reset()
    positions = [env.position]
    rewards = []
    states = [state]
    done = False

    # 축 고정
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.invert_yaxis()  # Y축을 위에서 아래로 표시
    ax.set_xticks(range(env.grid_size))
    ax.set_yticks(range(env.grid_size))
    ax.grid(True)

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
        action = q_values.argmax().item()
        
        next_state, reward, done = env.step(action)
        positions.append(env.position)
        rewards.append(reward)
        states.append(next_state)
        state = next_state
    
    for step in range(len(positions)):
        ax.clear()
        # 위험 영역(타원) 표시
        h, k = env.danger_ellipse['center']
        a, b = env.danger_ellipse['a'], env.danger_ellipse['b']
        ellipse = Ellipse(xy=(k, h), width=a, height=b, edgecolor='red', facecolor='red', alpha=0.3)
        ax.add_patch(ellipse)
        
        # 고정된 축 설정
        ax.set_xlim(0, env.grid_size)
        ax.set_ylim(0, env.grid_size)
        ax.invert_yaxis()
        ax.set_xticks(range(env.grid_size))
        ax.set_yticks(range(env.grid_size))
        ax.grid(True)

        # 경로 그리기
        current_positions = positions[:step+1]
        if current_positions:
            path_coords = np.array(current_positions)
            ax.plot(path_coords[:, 1], path_coords[:, 0], 'b-', alpha=0.5)
            for idx, (pos_x, pos_y) in enumerate(current_positions[:-1]):
                ax.plot(pos_y, pos_x, 'bo', alpha=0.3, markersize=8)

        # 시작점과 현재 위치
        ax.plot(env.start[1], env.start[0], 'go', markersize=15, label='Start')
        current_x, current_y = positions[step]
        ax.plot(current_y, current_x, 'r*', markersize=15, label='Current')

        # 누적 보상
        cumulative_reward = sum(rewards[:step]) if step > 0 else 0
        ax.set_title(f'Episode {episode_num+1}, Step {step}/{len(positions)-1}\n'
                     f'Position: {positions[step]}, Reward: {cumulative_reward:.1f}\n ' f'state: {states[step]}')
        ax.legend()
        
        plt.draw()
        plt.pause(0.2)





def train_dqn_with_visualization(env):
    state_dim = 8
    action_dim = 5  
    
    model = DQN(state_dim, action_dim).to(device)
    target_model = DQN(state_dim, action_dim).to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.MSELoss()

    replay_buffer = []
    max_buffer_size = 20000
    batch_size = 256
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.999
    epsilon_min = 0.01
    episodes = 2000
    target_update = 10
    
    # Visualization setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    rewards_history = []
    steps_history = []
    plt.ion()

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        episode_terminated = False
        
        # Run one episode
        while not done and not episode_terminated:
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model(state_tensor)
                action = q_values.argmax().item()
            
            next_state, reward, done = env.step(action)
            total_reward += reward
            step_count += 1
            
            # 누적 리워드가 -50 이하면 에피소드 종료
            if total_reward <= -50:
                episode_terminated = True
                done = True
            
            # Store in replay buffer
            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > max_buffer_size:
                replay_buffer.pop(0)

            state = next_state

            # Train model if buffer has enough samples
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
        steps_history.append(step_count)

        # Update plots
        ax1.clear()
        ax2.clear()

        ax1.plot(rewards_history, label='Rewards', color='blue')
        ax1.set_title("Episode Rewards")
        ax1.set_xlabel("Episodes")
        ax1.set_ylabel("Total Reward")
        ax1.legend()

        ax2.plot(steps_history, label='Steps', color='orange')
        ax2.set_title("Steps Per Episode")
        ax2.set_xlabel("Episodes")
        ax2.set_ylabel("Steps")
        ax2.legend()

        plt.draw()
        plt.pause(0.01)

        if (episode + 1) % 10 == 0:
            if not episode_terminated:  # Only visualize if episode wasn't terminated early
                visualize_episode_steps(env, model, episode, fig, ax1)
            avg_reward = sum(rewards_history[-10:]) / 10
            print(f"Episode {episode + 1}: Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}")
            if episode_terminated:
                print(f"Episode terminated early due to low reward: {total_reward}")

    plt.ioff()
    plt.show()

    return model, rewards_history





# 학습 실행
env = GridEnvironment()
trained_model = train_dqn_with_visualization(env)

'''class GridEnvironment:
    def __init__(self):
        self.grid_size = 10
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.start = (1, 1)  # 시작 지점.
        # 타원형 위험 영역 정의 (중심점, 장축, 단축)
        self.danger_ellipse = {'center': (5, 5), 'a': 2, 'b': 1}
        self.max_steps = 200
        self.current_steps = 0
        self.direction = 0  # 초기 방향 (0도)
        self.position = self.start
        self.last_angle = 0  # 이전 각도 저장
        self.reset()

    def reset(self):
        self.position = self.start
        self.direction = 0
        self.current_steps = 0
        self.last_angle = np.arctan2(0, 0)  # 초기 각도
        return self._get_state()
    
    def _is_in_ellipse(self, position):
        # 타원 내부인지 확인
        x, y = position
        h, k = self.danger_ellipse['center']
        a, b = self.danger_ellipse['a'], self.danger_ellipse['b']
        
        # 타원 방정식: ((x-h)/a)^2 + ((y-k)/b)^2 <= 1
        return ((x - h) / a) ** 2 + ((y - k) / b) ** 2 <= 1

    def _distance_to_ellipse(self, position):
        # 타원과의 대략적인 거리 계산 (간단한 근사)
        x, y = position
        h, k = self.danger_ellipse['center']
        a, b = self.danger_ellipse['a'], self.danger_ellipse['b']
        
        # 타원까지의 거리를 근사적으로 계산
        angle = np.arctan2(y - k, x - h)
        r = (a * b) / np.sqrt((b * np.cos(angle))**2 + (a * np.sin(angle))**2)
        actual_r = np.sqrt((x - h)**2 + (y - k)**2)
        return max(0, actual_r - r)

    def _get_state(self):
        # 각 방향의 위험 영역까지의 거리 계산
        directions = [-90, -45, 0, 45, 90]
        state_info = []
        for angle in directions:
            next_pos = self._calculate_position(angle)
            dist_to_danger = self._distance_to_ellipse(next_pos)
            state_info.append(dist_to_danger / self.grid_size)
        
        # 시작점과의 거리 정보
        dx = self.position[0] - self.start[0]
        dy = self.position[1] - self.start[1]
        dist_to_start = np.sqrt(dx**2 + dy**2) / self.grid_size
        
        # 시작점 기준 상대 각도 (-π ~ π 를 -1 ~ 1로 정규화)
        current_angle = np.arctan2(dy, dx) / np.pi
        
        # 현재 스텝 정보
        step_info = self.current_steps / self.max_steps
        
        return np.array(state_info + [dist_to_start, current_angle, step_info])

    def _calculate_position(self, angle):
        # 주어진 방향으로 이동했을 때의 위치 계산
        radians = np.radians(self.direction + angle)
        x, y = self.position
        dx = 0.5 * np.cos(radians)
        dy = 0.5 * np.sin(radians)
        new_x = np.clip(x + dx, 0, self.grid_size - 1)
        new_y = np.clip(y + dy, 0, self.grid_size - 1)
        return (new_x, new_y)
    
    def step(self, action):
        # 행동에 따라 회전 및 전진 처리
        rotations = [-90, -45, 0, 45, 90]
        self.direction = (self.direction + rotations[action]) % 360
        self.position = self._calculate_position(0)
        self.current_steps += 1
        
        # 현재 위치의 상대 각도 계산
        dx = self.position[0] - self.start[0]
        dy = self.position[1] - self.start[1]
        current_angle = np.arctan2(dy, dx)
        
        # 보상 계산
        reward = 0
        
        # 1. 생존 보상
        reward += 0.1
        
        # 2. 위험 영역 페널티
        if self._is_in_ellipse(self.position):
            return self._get_state(), -50.0, True
        
        # 3. 거리에 따른 보상
        dist_to_danger = self._distance_to_ellipse(self.position)
        reward += 0.5 * dist_to_danger
        
        # 4. 시작점과의 거리에 따른 페널티 (너무 멀어지지 않도록)
        dist_to_start = np.sqrt(dx**2 + dy**2)
        if dist_to_start > self.grid_size * 0.7:  # 격자 크기의 70% 이상 멀어지면 페널티
            reward -= 0.3
        
        # 5. 순회 진행도에 따른 보상
        angle_diff = (current_angle - self.last_angle) % (2 * np.pi)
        if angle_diff > 0:  # 반시계 방향 회전 보상
            reward += 0.2
        
        # 한바퀴 순회 체크 (시작점 근처로 돌아옴)
        if dist_to_start < 1.0 and self.current_steps > 50:  # 최소 50스텝 이상 진행
            reward += 10.0
            return self._get_state(), reward, True
        
        self.last_angle = current_angle
        done = self.current_steps >= self.max_steps
        
        return self._get_state(), reward, done'''

'''def train_dqn_with_visualization(env):
    state_dim = 8 
    action_dim = 5  
    
    model = DQN(state_dim, action_dim).to(device)
    target_model = DQN(state_dim, action_dim).to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.MSELoss()

    replay_buffer = []
    max_buffer_size = 20000
    batch_size = 256
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    episodes = 2000
    target_update = 10
    
    # 시각화를 위한 설정
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    rewards_history = []
    steps_history = []
    plt.ion()

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        # 에피소드 실행
        while not done:
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, done = env.step(action)
            total_reward += reward

            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > max_buffer_size:
                replay_buffer.pop(0)

            state = next_state

            # 학습 수행
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
        steps_history.append(env.current_steps)
        
        # 매 10 에피소드마다 시각화
        if (episode + 1) % 10 == 0:
            visualize_episode_steps(env, model, episode, fig, ax1)
            
            # 학습 진행 상황 그래프 업데이트
            ax2.clear()
            ax2.plot(rewards_history, label='Reward', alpha=0.7)
            ax2.plot(steps_history, label='Steps', alpha=0.7)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Value')
            ax2.legend()
            ax2.grid(True)
            ax2.set_title('Training Progress')
            
            plt.draw()
            plt.pause(0.01)
            
            avg_reward = sum(rewards_history[-10:]) / 10
            avg_steps = sum(steps_history[-10:]) / 10
            print(f"Episode {episode + 1}: Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.1f}, Epsilon: {epsilon:.2f}")

    plt.ioff()
    plt.show()
    return model'''

'''# 학습 실행
env = GridEnvironment()
trained_model, rewards = train_dqn(env)

# 보상 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(rewards)
plt.title('Training Rewards Over Time')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()'''