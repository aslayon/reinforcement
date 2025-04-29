import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import gc  # 가비지 컬렉션
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_episode(env, model, delay=0.5):
    state = env.reset()
    done = False
    
    plt.figure(figsize=(5,5))
    
    while not done:
        plt.imshow(env.render(), cmap="gray")  # 환경을 이미지로 렌더링
        plt.axis("off")
        plt.show(block=False)
        time.sleep(delay)
        plt.clf()
        
        # 모델로 액션 선택
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
        action = q_values.argmax().item()
        
        # 환경 업데이트
        state, _, done, _ = env.step(action)
    
    plt.imshow(env.render(), cmap="gray")  # 마지막 프레임 출력
    plt.axis("off")
    plt.show()

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
        
        self.small_goal_range = [(8, 8), (9, 9)]



        


        
    def reset(self):
        self.agent_pos = list(self.start_pos)
        self.position = self.agent_pos
        self.crashed = False
        self.current_steps = 0
        previous_distance = 0


        


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
        
        # 충돌 시 처리
        if self.crashed:
            return self.get_state(), -50, True, {}
        
        # 보상 계산: y = x 기준으로 보상 조정
        reward = 0
        x_start, y_start = self.start_pos

        # 그래프 아래: y < x -> 시작점에서 멀어질수록 보상 증가
        if y < x:
            if self.previous_distance < np.sqrt((x - x_start)**2 + (y - y_start)**2):
                reward += (np.sqrt((x - x_start)**2 + (y - y_start)**2) - self.previous_distance)*2
            else:
                reward -= 5
            

        # 그래프 위: y > x -> 시작점과 가까워질수록 보상 증가
        elif y > x:
            if self.previous_distance > np.sqrt((x - x_start)**2 + (y - y_start)**2):
                reward += (self.previous_distance - np.sqrt((x - x_start)**2 + (y - y_start)**2)  )*2
            else:
                reward -= 5

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
    
# 저장된 모델 불러오기
loaded_model = DQN(2, 4).to(device)
loaded_model.load_state_dict(torch.load("grid_dqn_model.pth", map_location=device))
loaded_model.eval()

# 시각화 실행
env = GridEnvironment()
visualize_episode(env, loaded_model)
