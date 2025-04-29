import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import gc  # 가비지 컬렉션
import time

goal_reached = 0 # 목표지점 도달 횟수

'''
입력값: x, y 위치 / heading(방향) / 초음파 센서 거리 (-45°, 0°, +45°)
출력값: 이동 방향 제어
    0: 우회전 (+45도)
    1: 전진
    2: 좌회전 (-45도)
'''

# ================================
# 그리드 좌표 매핑 (행, 열) = (y, x)
#     열 인덱스 →
# 행   0   1   2   3   4   5  ... 21
# ↓  -------------------------------
# 0 |(0,0)(0,1)(0,2)(0,3)(0,4)(0,5)...
# 1 |(1,0)(1,1)(1,2)(1,3)(1,4)(1,5)...
# 2 | ...
# 6 |(6,0)(6,1)(6,2)(6,3)(6,4)(6,5)...
# ================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class GridEnvironment:
    def __init__(self, grid_size=(7, 22)):
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size)
        self.grid[0, :] = 1 # 상단 벽
        self.grid[-1, :] = 1        # 하단 벽
        self.grid[:, 0] = 1       # 좌측 벽
        self.grid[:, -1] = 1    # 우측 벽

        self.grid[3,15] = 1 # 장애물 (3, 15)
        self.grid[4, 15] = 1 # 장애물 (3, 15)
        self.grid[3, 16] = 1 # 장애물 (3, 15)

        self.grid[2, 16] = 1 # 장애물 (4, 15)
        self.grid[4, 14] = 1 # 장애물 (4, 15)

        #self.grid[4:7, 4:7] = 1
        self.agent_pos = [3, 1] # 시작 위치
        self.start_pos = list(self.agent_pos)
        self.crashed = False
        self.current_steps = 0
        self.previous_distance = 0 # 이전 스텝에서의 거리.

        self.backstep = 0 # 뒤로 간 횟수


        self.heading = 0  # 각도도 (0도) 0도는 오른쪽, 90도는 위쪽, 180도는 왼쪽, 270도는 아래쪽
        
        # 시각화를 위한 속성 추가
        self.position = self.agent_pos
        self.start = self.start_pos
        self.prevPosition = []
        self.small_goal_range = [(1, 20), (5, 20)]
        self.direction_map = {
        0:   (0, 1),     # →
        45:  (-1, 1),    # ↗ (위로 한 칸, 오른쪽 한 칸)
        90:  (-1, 0),    # ↑
        135: (-1, -1),   # ↖
        180: (0, -1),    # ←
        225: (1, -1),    # ↙
        270: (1, 0),     # ↓
        315: (1, 1)      # ↘
    }

    def reset(self):
        self.agent_pos = list(self.start_pos)
        self.position = self.agent_pos
        self.heading = 0
        self.crashed = False
        self.current_steps = 0
        previous_distance = 0
        self.minus_count = 0
        self.prevPosition = []
        self.backstep = 0 # 뒤로 간 횟수


        return self.get_state()
        
    def dist_to_goal(self, x, y):
        # 목표 지점과의 거리 계산 (예: 유클리드 거리)
        # 작은 목표 지점 (1, 20) ~ (5, 20) 사이의 거리 계산
        min_dist = float('inf')
        for gx in range(self.small_goal_range[0][0], self.small_goal_range[1][0] + 1):
            gy = self.small_goal_range[0][1]  # y는 고정: 20
            dist = np.sqrt((x - gx) ** 2 + (y - gy) ** 2)
            if dist < min_dist:
                min_dist = dist
        return min_dist
    
    # 현재 위치, 목표지점까지의 거리, 현재 각도 , 세 방향의 거리(센서 결과) ( x,y, 각도 (-45, 0, 45) )  _ 대각선의 거리도 1로 가정(루트2 가 아님)
    def get_sensor_distances(self, x, y, heading):
    # 센서가 바라보는 3방향: -45도, 0도, +45도
        angles = [(-90 + heading) % 360,(-45 + heading) % 360, heading % 360, (45 + heading) % 360,(90 + heading) % 360]
        

        distances = []
        for angle in angles:
            angle = int(angle) % 360
            angle = (round(angle / 45) * 45) % 360  # 45도로 라운딩 후 정규화
            dx, dy = self.direction_map[angle]
            dist = 0
            nx, ny = x, y

            # 벽(1)을 만날 때까지 이동
            while True:
                nx += dx
                ny += dy
                dist += 1
                if nx < 0 or nx >= self.grid_size[0] or ny < 0 or ny >= self.grid_size[1]:
                    break  # 그리드 밖으로 나가면 중단
                if self.grid[nx, ny] == 1:
                    break  # 벽 만나면 중단
            distances.append(dist)

        return distances  # [왼쪽(-45도), 정면(0도), 오른쪽(45도)]


        

    
    def is_in_goal(self, x, y):
        goal_x_min = self.small_goal_range[0][0]
        goal_x_max = self.small_goal_range[1][0]
        goal_y = self.small_goal_range[0][1]
        return goal_x_min <= x <= goal_x_max and y == goal_y    
    
    def step(self, action):
        self.current_steps += 1
        x, y = self.agent_pos
        
        # 이동 로직
        if action == 0:  # +45도 회전
            self.heading = (self.heading + 45) % 360
        elif action == 1:  # 전진
            # heading 값을 45도 간격으로 정규화
            normalized_heading = (round(self.heading / 45) * 45) % 360
            if normalized_heading not in self.direction_map:
                
                normalized_heading = 0  # 기본값 설정
                
            dx, dy = self.direction_map[normalized_heading]
            new_x = x + dx
            new_y = y + dy

            if self.grid[new_x, new_y] == 0:
                x, y = new_x, new_y
            else:
                self.crashed = True
        elif action == 2:  # -45도 회전
            self.heading = (self.heading - 45) % 360

        # 디버깅 정보 출력
        #print(f"Action: {action}, Heading: {self.heading}, Position: {x, y}")

        
        self.agent_pos = [x, y]
        
        self.position = list(self.agent_pos)

        reward = -0.1

        # 충돌 시 처리
        if self.crashed:
            return self.get_state(), reward, True, {}
        
        # 최근 방문 위치 업데이트 (최대 5개까지만 유지)
        if len(self.prevPosition) >= 5:
            self.prevPosition.pop(0)
        self.prevPosition.append((x, y))

        # 최근 5번 중에 해당 위치가 3번 이상 등장하면 종료
        if self.prevPosition.count((x, y)) >= 3:
            return self.get_state(), reward, True, {}


        
        reward = 1


        sensor_distances = self.get_sensor_distances(x, y, self.heading)
        front_distance = sensor_distances[1]    # 정면 거리

        #print(sensor_distances)

        if front_distance < 2:
            reward -= 2

        # 목표 지점에 도달했는지 확인 (예: (5, 20) 위치)   
        if self.is_in_goal(x, y):
            reward = 30
            global goal_reached
            goal_reached += 1
            return self.get_state(), reward, True, {}
        

        

        if self.current_steps > 100:
            return self.get_state(), -1, True, {}


        

        return self.get_state(), reward, False, {}
    

    # 현재 위치,현재 각도 , 세 방향의 거리(센서 결과) ( x,y, 각도 (-45, 0, 45) )
    def get_state(self):
        # heading → one-hot 인코딩 (0 ~ 315 → 8방향)
        heading_idx = self.heading // 45
        heading_onehot = [0] * 8
        heading_onehot[heading_idx] = 1

        # 상태: [x, y, heading_onehot(8개), 센서 3개]
        state = list(self.get_sensor_distances(self.agent_pos[0], self.agent_pos[1], self.heading))
        return np.array(state, dtype=np.float32)

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
 
def visualize_episode_steps(env, model, episode_num, fig=None, ax=None):
    """한 에피소드의 스텝별 이동 과정을 시각화"""
    state = env.reset()
    done = False
    path = [env.agent_pos.copy()]  # 시작 위치 저장
    rewards = []
    headings = [env.heading]  # 시작 heading 저장

    # 에피소드 전체 경로 미리 계산
    while not done and env.current_steps < 100:  # 무한 루프 방지
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
        action = q_values.argmax().item()
        next_state, reward, done, _ = env.step(action)

        path.append(env.agent_pos.copy())
        headings.append(env.heading)  # ✅ 스텝마다 heading 저장
        rewards.append(reward)
        state = next_state

    # 각 스텝별로 시각화
    for step in range(len(path)):
        ax[0].clear()  # 첫 번째 subplot만 초기화

        # 그리드 초기화
        grid_display = np.zeros(env.grid_size)
        for i in range(env.grid_size[0]):
            for j in range(env.grid_size[1]):
                if env.grid[i, j] == 1:
                    grid_display[i, j] = 0.7  # 벽/장애물

        # 위험 영역 표시
        x1, y1 = env.small_goal_range[0]
        x2, y2 = env.small_goal_range[1]
        grid_display[x1:x2+1, y1:y2+1] = 0.3
        ax[0].imshow(grid_display, cmap='Greys', alpha=0.5)

        # 현재까지의 경로 표시
        if step > 0:
            path_coords = np.array(path[:step+1])
            ax[0].plot(path_coords[:, 1], path_coords[:, 0], 'b-', alpha=0.7)

            # 이동 경로상의 모든 점 표시
            for idx, (pos_x, pos_y) in enumerate(path[:step]):
                ax[0].plot(pos_y, pos_x, 'bo', alpha=0.3, markersize=8)

        # 시작점과 현재 위치 표시
        ax[0].plot(path[0][1], path[0][0], 'go', markersize=15, label='Start')
        current_x, current_y = path[step]
        ax[0].plot(current_y, current_x, 'r*', markersize=15, label='Current')

        # 🔄 현재 heading 방향 표시 (스텝에 맞게)
        heading = headings[step] if step < len(headings) else env.heading
        dx, dy = env.direction_map[heading]
        ax[0].arrow(current_y, current_x, dy * 0.5, dx * 0.5,
                    head_width=0.3, head_length=0.3, fc='red', ec='red')

        # 격자 그리기
        for i in range(env.grid_size[0]):
            for j in range(env.grid_size[1]):
                ax[0].axhline(y=i+0.5, color='black', linewidth=0.5, alpha=0.3)
                ax[0].axvline(x=j+0.5, color='black', linewidth=0.5, alpha=0.3)

        ax[0].grid(False)
        ax[0].set_xticks(range(env.grid_size[1]))
        ax[0].set_yticks(range(env.grid_size[0]))

        # 현재까지의 누적 보상 계산
        cumulative_reward = sum(rewards[:step]) if step > 0 else 0
        ax[0].set_title(f'Episode {episode_num+1}, Step {step}/{len(path)-1}\n'
                        f'Position: {path[step]}, Heading: {heading}°, Reward: {cumulative_reward:.1f}')
        ax[0].legend()

        plt.draw()
        plt.pause(0.05)  # 지연 시간 감소

    # 메모리 정리
    gc.collect()
    return path, rewards  # 경로와 보상 반환


def train_dqn_with_visualization(env, max_episodes=3000):
    state_dim = len(env.get_state())  # 상태 차원
    action_dim = 3  # 행동 차원 (-45 도, 전진 , 45도)
    
    model = DQN(state_dim, action_dim).to(device)
    target_model = DQN(state_dim, action_dim).to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.MSELoss()

    replay_buffer = []
    max_buffer_size = 20000
    batch_size = 256
    gamma = 0.99
    # 입실론 감소 일정을 더 세심하게 조정
    epsilon = 1.0
    epsilon_min = 0.3
    epsilon_decay = 0.9995  
    target_update = 10
    
    # 시각화를 위한 설정
    plt.ion()
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    
    rewards_history = []
    steps_history = []
    
    # 주기적으로 그림 저장
    save_interval = 25  # 25 에피소드마다 저장

    for episode in range(max_episodes):
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

            next_state, reward, done, _ = env.step(action)
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

        # 그리고 훈련 루프에서:
        epsilon = max(epsilon * epsilon_decay, epsilon_min) 
        
        if episode % target_update == 0:
            target_model.load_state_dict(model.state_dict())

        rewards_history.append(total_reward)
        steps_history.append(env.current_steps)
        
        # 매 N 에피소드마다 시각화 (N을 증가시켜 부하 감소)
        visualization_interval = 1000  # 20 에피소드마다 시각화
        if (episode + 1) % visualization_interval == 0:
            # 학습 진행 상황 그래프 업데이트 (오른쪽 그래프만)
            ax[1].clear()
            ax[1].plot(rewards_history, label='Reward', alpha=0.7)
            ax[1].plot(steps_history, label='Steps', alpha=0.7)
            ax[1].set_xlabel('Episode')
            ax[1].set_ylabel('Value')
            ax[1].legend()
            ax[1].grid(True)
            ax[1].set_title('Training Progress')
            
            # 먼저 그래프만 그리기
            plt.draw()
            plt.pause(0.01)
            
            # 그 다음 에피소드 시각화 (메모리 효율성을 위해)
            path, path_rewards = visualize_episode_steps(env, model, episode, fig, ax)
            
            avg_reward = sum(rewards_history[-10:]) / 10
            avg_steps = sum(steps_history[-10:]) / 10
            print(f"Episode {episode + 1}: Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.1f}, Epsilon: {epsilon:.2f}")
            
            # 주요 에피소드마다 이미지 저장
            if (episode + 1) % save_interval == 0:
            
                
                # 여기에 메모리 관리 코드 추가
                plt.close('all')  # 모든 그림 닫기
                plt.figure()  # 새 그림 생성
                plt.close()  # 바로 닫기
                
                # 새 그림 다시 만들기
                fig, ax = plt.subplots(1, 2, figsize=(16, 8))
                plt.ion()
                
                # 강제 가비지 컬렉션
                gc.collect()
                
                # 잠시 쉬기
                time.sleep(0.5)
        
        # 출력만 더 자주하기
        elif (episode + 1) % 100 == 0:
            avg_reward = sum(rewards_history[-10:]) / 10
            avg_steps = sum(steps_history[-10:]) / 10
            print(f"Episode {episode + 1}: Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.1f}, Epsilon: {epsilon:.2f}")

    # 학습 종료 후 처리
    plt.ioff()
    plt.close('all')
    
    # 학습된 모델 저장
    torch.save(model.state_dict(), "직선으로이동_장애물도_센서값만입력력.pth")
    print("학습 완료 및 모델 저장됨.")
    
    return model

# 학습 실행
if __name__ == "__main__":
    env = GridEnvironment()
    trained_model = train_dqn_with_visualization(env)
    print(f"Total goal reached: {goal_reached} times")