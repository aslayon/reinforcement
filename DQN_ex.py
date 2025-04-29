import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32
memory_size = 10000
num_episodes = 100

# Neural Network for DQN

# 입력은 [x, x_dot, theta, theta_dot] 즉 4
# 출력은 좌, 우 2 를 선택했을 때의 Q 
# TD 계산법. 현재 상태에서 a 를 선택했을 때의 기대 보상. 무수히 많은 S 가 존재하기 때문에 회귀 방정식을 구하는 과정을 신경망으로 구현.
#  Q 값은 해당 선택을 했을 때 얻는 보상(r) _ 직관적 보상   +   미래의 기대되는 보상 (예측 보상( 이후 가능한 액션들의 보상들의 합. 해당 값들은 감마(감가율) 을 곱함 예시 _ 4스텝뒤에 보상을 받는다 -> 감마^4))
#  이후의 모든 상태와 행동은 확률적임(랜덤성이 짙어 확률적으로 계산함)
# 따라서 가장 높은 Q 값을 대표값으로 선택 ( 그리디 정책 )_타깃 폴리시
# 타깃 정책이 그리디 하지만 행동정책은 다르게 설정 -> 새로운 경험을 더 많이 하고, 복기가 가능 -> 성능 향상
# 이렇게 신경망을 통해 예측한 Q 값과 실재 탐험하며 얻은 샘플값과의 차를 반영하여 가중치 업데이트

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Initialize environment and Q-network
env = gym.make('CartPole-v1', render_mode='human')  # Enable human rendering
q_network = DQN(env.observation_space.shape[0], env.action_space.n)
target_network = DQN(env.observation_space.shape[0], env.action_space.n)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
memory = ReplayMemory(memory_size)

# Function to select action using epsilon-greedy policy
def select_action(state, epsilon):
    if random.random() < epsilon:
        action = env.action_space.sample()  # Random action (exploration)
        print("Exploration: Random action selected")
        exploration = True
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = q_network(state_tensor)
        action = q_values.argmax().item()  # Greedy action (exploitation)
        print(f"Exploitation: Greedy action selected with Q-values: {q_values.detach().numpy()}")
        exploration = False
    return action, exploration

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0

    for t in range(200):
        env.render()  # Render the environment at each step
        
        action, exploration = select_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)

        # Print Q values, state, action, and exploration status
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = q_network(state_tensor)
        print(f"Episode {episode}, Step {t}")
        print(f"State: {state}")
        print(f"Action: {action} {'(Exploration)' if exploration else '(Exploitation)'}")
        print(f"Q-values: {q_values.detach().numpy()}")
        print("------")

        memory.push((state, action, reward, next_state, done))
        total_reward += reward
        state = next_state

        if done:
            print(f"Episode {episode} ended after {t + 1} steps")
            break

        # Train the Q-network if enough samples are available in memory
        if len(memory) >= batch_size:
            batch = memory.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)

            # Compute TD target
            with torch.no_grad():
                next_q_values = target_network(next_states).max(1, keepdim=True)[0]
                td_target = rewards + gamma * next_q_values * (1 - dones)

            # Compute current Q values
            current_q_values = q_network(states).gather(1, actions)

            # Compute loss and update the Q-network
            loss = nn.MSELoss()(current_q_values, td_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Update epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Update target network periodically
    if episode % 10 == 0:
        target_network.load_state_dict(q_network.state_dict())

    print(f"Episode {episode} finished with total reward: {total_reward}")

env.close()


#상태는 [위치, 속도, 각도, 각속도]
#Q-values: [[0.012, -0.003]]
#첫 번째 값은 왼쪽으로 이동했을 때의 Q 값, 두 번째 값은 오른쪽으로 이동했을 때의 Q 값