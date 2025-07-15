import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import gc  # 가비지 컬렉션
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class GridEnvironment:
    def __init__(self, grid_size=(11, 11)):
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size)
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        self.grid[4:7, 4:7] = 1
        self.agent_pos = [1, 1]
        self.start_pos = list(self.agent_pos)
        self.crashed = False
        self.current_steps = 0
        self.previous_distance = 0 # 이전 스텝에서의 거리.
        # 시각화를 위한 속성 추가
        self.position = self.agent_pos
        self.start = self.start_pos
        self.prevPosition = []
        self.small_goal_range = [(8, 8), (9, 9)]



        


        
    def reset(self):
        self.agent_pos = list(self.start_pos)
        self.position = self.agent_pos
        self.crashed = False
        self.current_steps = 0
        previous_distance = 0
        self.minus_count = 0
        self.prevPosition = []
        


        return np.array(self.agent_pos)
    
    def step(self, action):
        self.current_steps += 1
        x, y = self.agent_pos
        
        # 이동 로직
        if action == 0:  # 상
            if x > 0:
                x -= 1
            else:
                self.crashed = True
        elif action == 1:  # 하
            if x < self.grid_size[0] - 1:
                x += 1
            else:
                self.crashed = True
        elif action == 2:  # 좌
            if y > 0:
                y -= 1
            else:
                self.crashed = True
        elif action == 3:  # 우
            if y < self.grid_size[1] - 1:
                y += 1
            else:
                self.crashed = True

        # 새로운 좌표가 '1'인 접근 불가 구역이면 충돌
        if self.grid[x, y] == 1:
            self.crashed = True
        
        self.agent_pos = [x, y]
        self.position = self.agent_pos
        reward = 0
        # 충돌 시 처리
        if self.crashed:
            return self.get_state(), reward, True, {}
        
         # 최근 방문 위치 업데이트 (최대 4개까지만 유지)
        if len(self.prevPosition) >= 4:
            self.prevPosition.pop(0)
        self.prevPosition.append((x, y))

        # 최근 4번 이내에 방문한 적 있으면 종료
        if (x, y) in self.prevPosition[:-1]:  # 방금 이동한 위치 제외하고 체크
            return self.get_state(), -50, True, {}

        # 보상 계산: y = x 기준으로 보상 조정
        
        x_start, y_start = self.start_pos

        # 그래프 아래: y < x -> 시작점에서 멀어질수록 보상 증가
        if y < x:
            if self.previous_distance < np.sqrt((x - x_start)**2 + (y - y_start)**2):
                reward += (np.sqrt((x - x_start)**2 + (y - y_start)**2) - self.previous_distance)*2
            else:
                if self.minus_count > 1:
                    return self.get_state(),  reward, True, {}
                reward -= 10
                self.minus_count += 1
            

        # 그래프 위: y > x -> 시작점과 가까워질수록 보상 증가
        elif y > x:
            if self.previous_distance > np.sqrt((x - x_start)**2 + (y - y_start)**2):
                reward += (self.previous_distance - np.sqrt((x - x_start)**2 + (y - y_start)**2)  )*2
            else:
                if self.minus_count > 1:
                    return self.get_state(), reward, True, {}
                reward -= 10
                self.minus_count += 1
                

        self.previous_distance = np.sqrt((x - x_start)**2 + (y - y_start)**2)


        

        return self.get_state(), reward, False, {}

    def get_state(self):
        return np.array(self.agent_pos)

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
    
    # 에피소드 전체 경로 미리 계산
    while not done and env.current_steps < 100:  # 무한 루프 방지
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
        action = q_values.argmax().item()
        next_state, reward, done, _ = env.step(action)
        path.append(env.agent_pos.copy())
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
                    f'Position: {path[step]}, Reward: {cumulative_reward:.1f}')
        ax[0].legend()
        
        plt.draw()
        plt.pause(0.05)  # 지연 시간 감소
    
    # 메모리 정리
    gc.collect()
    return path, rewards  # 경로와 보상 반환

# 환경 및 모델 초기화
env = GridEnvironment()
model = DQN(state_dim=2, action_dim=4).to(device)  # 상태는 (x, y), 행동은 4가지

# 저장된 가중치 불러오기
model.load_state_dict(torch.load("grid_dqn_model.pth", map_location=device)) # 모델 가중치 불러오기
model.eval()  # 평가 모드로 설정


# 시각화용 figure 준비
plt.ion()
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

# 테스트 에피소드 시각화 
for episode in range(5): 
    path, rewards = visualize_episode_steps(env, model, episode_num=episode, fig=fig, ax=ax)
    print(f"[Test Episode {episode+1}] 총 스텝: {len(path)} | 총 보상: {sum(rewards):.2f}")
    plt.pause(1)  
    
plt.ioff()
plt.show()