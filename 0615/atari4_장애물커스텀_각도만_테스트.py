import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import gc  # 가비지 컬렉션
import time
from collections import deque

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
        success = self.generate_valid_map()
        if not success:
            print("⚠️ 경로 생성 실패 - 예비 맵 사용됨")

        self.agent_pos = list(self.start_pos)
        self.position = self.agent_pos
        self.heading = 0
        self.crashed = False
        self.current_steps = 0
        self.prevPosition = []
        self.backstep = 0

        return self.get_state()

    def place_random_obstacles(self, count=30, avoid_area=None): # 장애물 배치 함수 장애물은 랜덤으로
        placed = 0
        avoid_set = set(avoid_area or [])
        while placed < count:
            x = random.randint(1, self.grid_size[0] - 2)
            y = random.randint(1, self.grid_size[1] - 2)
            if self.grid[x, y] == 0 and (x, y) not in avoid_set:
                self.grid[x, y] = 1
                placed += 1

    def is_path_exists(self, grid, start, goal_cells): # 클래스 내부로 이동
        """경로가 존재하는지 확인하는 함수 (BFS 사용)"""
        visited = set()
        queue = deque([tuple(start)])
        visited.add(tuple(start))

        while queue:
            x, y = queue.popleft()
            if (x, y) in goal_cells:
                return True
            
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and 
                    grid[nx, ny] == 0 and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False

    def generate_valid_map(self, max_trials=20, obstacle_count_range=(10, 40)):
        for trial in range(max_trials):
            # 그리드 초기화
            self.grid = np.zeros(self.grid_size)
            self.grid[0, :] = 1  # 상하좌우 벽
            self.grid[-1, :] = 1
            self.grid[:, 0] = 1
            self.grid[:, -1] = 1

            # 피해야 할 위치들 (시작점 + 목표지점 전체)
            avoid = [tuple(self.start_pos)] + [(x, 20) for x in range(1, 6)]

            # 장애물 배치
            count = random.randint(*obstacle_count_range)
            self.place_random_obstacles(count=count, avoid_area=avoid)

            # 경로 존재 확인
            goal_cells = [(x, 20) for x in range(1, 6)]
            if self.is_path_exists(self.grid, self.start_pos, goal_cells):
                #print(f"✅ 유효한 맵 생성됨 (시도 {trial + 1}/{max_trials}, 장애물 {count}개)")
                return True
        
        print(f"❌ {max_trials}번 시도 후 유효한 맵 생성 실패")
        return False

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
    
    def get_sensor_distances(self, x, y, heading):
        """센서가 바라보는 3방향: -45도, 0도, +45도의 거리 반환"""
        angles = [
                (heading - 90) % 360,
                (heading - 45) % 360,
                heading % 360,
                (heading + 45) % 360,
                (heading + 90) % 360
        ]
        
        distances = []
        for angle in angles:
            angle = int(angle) % 360
            angle = (round(angle / 45) * 45) % 360  # 45도로 라운딩 후 정규화
            
            if angle not in self.direction_map:
                angle = 0  # 기본값
                
            dx, dy = self.direction_map[angle]
            dist = 0
            nx, ny = x, y

            # 벽(1)을 만날 때까지 이동
            while True:
                nx += dx
                ny += dy
                dist += 1
                if (nx < 0 or nx >= self.grid_size[0] or 
                    ny < 0 or ny >= self.grid_size[1] or
                    self.grid[nx, ny] == 1):
                    break
            distances.append(dist)

        return distances  # [왼쪽(-45도), 정면(0도), 오른쪽(+45도)]
    
    def is_in_goal(self, x, y):
        goal_x_min = self.small_goal_range[0][0]
        goal_x_max = self.small_goal_range[1][0]
        goal_y = self.small_goal_range[0][1]
        return goal_x_min <= x <= goal_x_max and y == goal_y    
    
    def step(self, action):
        self.current_steps += 1
        x, y = self.agent_pos
        
        reward = -0.1

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
            
            # 범위 체크 추가
            if (0 <= new_x < self.grid_size[0] and 
                0 <= new_y < self.grid_size[1] and 
                self.grid[new_x, new_y] == 0):
                x, y = new_x, new_y
            else:
                self.crashed = True
        elif action == 2:  # -45도 회전
            self.heading = (self.heading - 45) % 360
        
        self.agent_pos = [x, y]
        self.position = list(self.agent_pos)

        

        # 충돌 시 처리
        if self.crashed:
            return self.get_state(), -10, True, {}
        
        '''# 최근 방문 위치 업데이트 (최대 8개까지만 유지)
        if len(self.prevPosition) >= 8:
            self.prevPosition.pop(0)
        self.prevPosition.append((x, y))

        # 최근 5번 중에 해당 위치가 5번 이상 등장하면 종료
        if self.prevPosition.count((x, y)) >= 5:
            return self.get_state(), -5, True, {}
'''
        reward = 1

        sensor_distances = self.get_sensor_distances(x, y, self.heading)
        front_distance = sensor_distances[1]    # 정면 거리

        if front_distance < 2:
            reward -= 2

        # 목표 지점에 도달했는지 확인   
        if self.is_in_goal(x, y):
            reward = 30
            global goal_reached
            #goal_reached += 1
            return self.get_state(), reward, True, {}

        if self.current_steps > 100: # 도달하지 못했을 때
            return self.get_state(), -30, True, {}

        return self.get_state(), reward, False, {}
    
    def get_state(self):
        # 센서 거리만 반환 (3개 값)
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

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 환경 및 모델 초기화
env = GridEnvironment()
state_dim = len(env.get_state())  # 상태 차원
action_dim = 3  # 0: 우회전, 1: 전진, 2: 좌회전

# 모델 정의 및 가중치 로딩
model = DQN(state_dim, action_dim).to(device)
model.load_state_dict(torch.load("직선으로이동_장애물도_센서값만입력.pth", map_location=device))
model.eval()

# 시각화용 Figure 초기화
plt.ion()
fig, ax = plt.subplots(1, 2, figsize=(16, 8))  # [0]: 맵, [1]: 보상 그래프

# 테스트 에피소드 실행 및 시각화
for episode in range(5):
    path, rewards = visualize_episode_steps(env, model, episode_num=episode, fig=fig, ax=ax)
    print(f"[Test Episode {episode + 1}] 총 스텝: {len(path)} | 총 보상: {sum(rewards):.2f}")
    plt.pause(1.0)  # 에피소드 간 pause

plt.ioff()
plt.show()
