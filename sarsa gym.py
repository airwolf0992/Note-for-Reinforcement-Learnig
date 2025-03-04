import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pygame
import matplotlib.pyplot as plt

# 定义 Q 网络
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 定义 SARSA 代理
class SARSAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = np.array(state)
        if len(state.shape) != 2 or state.shape[0] != 1:
            raise ValueError("State should be a 2D array with shape (1, state_size).")
        state = torch.FloatTensor(state)
        act_values = self.model(state)
        action = np.argmax(act_values.detach().numpy())
        return action
    
    def update(self, state, action, reward, next_state, next_action, done):
        # 将状态和下一个状态转换为numpy数组并指定数据类型
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        
        # 将numpy数组转换为PyTorch张量
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        
        # 获取当前状态-动作对的Q值
        current_q = self.model(state)[0][action]
        
        # 获取下一个状态-动作对的Q值，如果回合结束则为0
        next_q = self.model(next_state)[0][next_action] if not done else torch.tensor(0.0, dtype=torch.float32)
        
        # 计算目标Q值：即时奖励 + 折扣因子 * 下一个Q值
        target = torch.tensor(reward, dtype=torch.float32) + self.gamma * next_q
        
        # 计算当前Q值和目标Q值之间的均方误差损失
        loss = self.criterion(current_q, target.detach())
        
        # 梯度清零
        self.optimizer.zero_grad()
        
        # 反向传播计算梯度
        loss.backward()
        
        # 更新网络参数
        self.optimizer.step()
        
        # 衰减探索率，但保持不低于最小值
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    EPISODES = 500
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    font = pygame.font.SysFont("arial", 20)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = SARSAgent(state_size, action_size)
    scores_list = []
    
    for e in range(EPISODES):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        action = agent.act(state)
        for time in range(1000):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    quit()
            frame = env.render()
            if frame is not None:
                surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                text_surface = font.render(
                    f"Episode: {e+1} Score: {time}", True, (255, 0, 0)
                )
                scaled_surface = pygame.transform.scale(surface, (800, 600))
                scaled_surface.blit(text_surface, (10, 10))
                screen.blit(scaled_surface, (0, 0))
                pygame.display.flip()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward = reward if not done else -10
            next_state = np.array(next_state, dtype=np.float32).flatten()
            next_state = np.reshape(next_state, [1, state_size])
            next_action = agent.act(next_state)
            agent.update(state, action, reward, next_state, next_action, done)
            state = next_state
            action = next_action
            if done:
                print(
                    f"Episode: {e + 1}/{EPISODES}, Score: {time}, Epsilon: {agent.epsilon:.2f}"
                )
                break
            scores_list.append(time)
        if e % 100 == 0:
            torch.save(agent.model.state_dict(), f"sarsa_model_{e}.pth")
    
    plt.plot(scores_list)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Scores')
    plt.show()
    env.close()
    pygame.quit()