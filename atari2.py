import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import gc  # 가비지 컬렉션
import time

'''
입력값 x,y 좌표
출력값: 상하좌우 이동
상: 0, 하: 1, 좌: 2, 우: 3
'''

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

def train_dqn_with_visualization(env, max_episodes=5000):
    state_dim = len(env.get_state())  # 상태 차원
    action_dim = 4  # 상, 하, 좌, 우
    
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
    epsilon_min = 0.1
    epsilon_decay = 0.999  
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
    torch.save(model.state_dict(), "grid_dqn_model.pth")
    print("학습 완료 및 모델 저장됨.")
    
    return model

# 학습 실행
if __name__ == "__main__":
    env = GridEnvironment()
    trained_model = train_dqn_with_visualization(env)