import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pygame
import matplotlib.pyplot as plt
# 定义 Q 网络
# 这是一个用于近似 Q 值函数的神经网络，输入为状态，输出为每个动作对应的 Q 值
class QNetwork(nn.Module):
  def __init__(self, state_size, action_size):
    # 调用父类 nn.Module 的构造函数
    super(QNetwork, self).__init__()
    # 第一个全连接层，将输入的状态向量映射到 64 维的隐藏层
    self.fc1 = nn.Linear(state_size, 64)
    # 第二个全连接层，将 64 维的隐藏层映射到另一个 64 维的隐藏层
    self.fc2 = nn.Linear(64, 128)
    # 第三个全连接层，将 64 维的隐藏层映射到动作空间的维度，输出每个动作的 Q 值
    self.fc3 = nn.Linear(128, action_size)
  def forward(self, x):
    # 通过第一个全连接层和ReLU激活函数
    x = torch.relu(self.fc1(x))
    # 通过第二个全连接层和ReLU激活函数
    x = torch.relu(self.fc2(x))
    # 通过第三个全连接层得到最终的 Q 值输出
    return self.fc3(x)
# 定义 DQN 代理 # 该类封装了 DQN 算法的核心逻辑，包括经验回放、动作选择和网络更新等
class DQNAgent:
  def __init__(self, state_size, action_size):
    # 状态空间的维度
    self.state_size = state_size
    # 动作空间的维度
    self.action_size = action_size
    # 经验回hfill缓冲区，使用 deque 存储智能体的经验，最大容量为 2000
    self.memory = deque(maxlen=10000)
    # 折扣因子，用于平衡即时奖励和未来奖励的重要性
    self.gamma = 0.9
    # 初始探索率，控制智能体进行随机探索的概率
    self.epsilon = 0.2
    # 最小探索率，探索率不会低于该值
    self.epsilon_min = 0.01
    # 探索率衰减率，每次回放后探索率会乘以该值逐渐降低
    self.epsilon_decay = 0.995
    # 学习率，控制网络参数更新的步长
    self.learning_rate = 0.001
    # 实例化 Q 网络
    self.model = QNetwork(state_size, action_size)
    # 使用 Adam 优化器来更新 Q 网络的参数
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    # 定义均方误差损失函数，用于计算预测 Q 值和目标 Q 值之间的误差
    self.criterion = nn.MSELoss()
  # 将智能体的经验存储到经验回hfill缓冲区中
  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
  # 根据当前状态选择动作
  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)
    # 确保状态是 NumPy 数组且是一维的
    state = np.array(state)
    # 检查状态数组是否为二维且第一维度长度为1
    if len(state.shape) != 2 or state.shape[0] != 1:
      # 如果状态数组不符合要求，抛出错误提示
      raise ValueError("State should be a 2D array with shape (1, state_size).")
    # 将状态数组转换为FloatTensor类型，以适应深度学习模型的输入要求
    state = torch.FloatTensor(state)
    # 使用当前模型对状态进行评估，得到各个动作的价值
    act_values = self.model(state)
    # 选择具有最高动作价值的动作
    action = np.argmax(act_values.detach().numpy())
    # 返回选择的动作
    return action

  # 经验回放过程
  def replay(self, batch_size):
    # 如果经验回hfill缓冲区中的数据量小于批量大小，则直接返回
    if len(self.memory) < batch_size:
      return
    # 从经验回hfill缓冲区中随机采样一个批量数据
    minibatch = random.sample(self.memory, batch_size)
    # 遍历批量中的每个经验样本
    for state, action, reward, next_state, done in minibatch:
      # 将状态和下一个状态转换为numpy数组，并保持二维形状
      state = np.array(state, dtype=np.float32)
      next_state = np.array(next_state, dtype=np.float32)
      # 将状态和下一个状态转换为PyTorch张量
      state = torch.FloatTensor(state)
      next_state = torch.FloatTensor(next_state)
      # 初始化目标值为当前奖励
      target = reward
      # 如果当前状态不是终止状态，则计算目标Q值
      if not done:
        # SARSA算法使用当前策略选择下一个动作
        next_action = self.act(next_state.numpy())
        next_state_tensor = torch.FloatTensor(next_state)
        next_q_value = self.model(next_state_tensor)[0][next_action]
        target = reward + self.gamma * next_q_value.item()
      # 获取当前状态的预测Q值
      target_f = self.model(state)
      # 更新所选动作对应的Q值
      target_f[0][action] = target
      # 清空优化器的梯度
      self.optimizer.zero_grad()
      # 计算预测Q值和目标Q值之间的损失
      loss = self.criterion(self.model(state), target_f)
      # 反向传播计算梯度
      loss.backward()
      # 更新网络参数
      self.optimizer.step()
    # 如果当前探索率大于最小探索率，则衰减探索率
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
  # 设置总训练轮数
  EPISODES = 500
  # 创建CartPole环境，设置渲染模式为rgb_array以获取像素数据
  env = gym.make("CartPole-v1", render_mode="rgb_array")
  # 初始化pygame
  pygame.init()
  # 创建pygame显示窗口，设置窗口尺寸为800x600
  screen = pygame.display.set_mode((800, 600))
  # 设置字体样式和大小
  font = pygame.font.SysFont("arial", 20)
  # 获取状态空间的维度
  state_size = env.observation_space.shape[0]
  # 获取动作空间的维度
  action_size = env.action_space.n
  # 创建DQN智能体
  agent = DQNAgent(state_size, action_size)
  # 设置批量大小
  batch_size = 128
  # 初始化分数列表，用于记录每轮的得分
  scores_list = []
  # 开始训练循环
  for e in range(EPISODES):
    # 重置环境，获取初始状态
    state, _ = env.reset()
    # 将状态reshape为适合网络的形状
    state = np.reshape(state, [1, state_size])
    # 单轮训练循环
    for time in range(1000):
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          env.close()
          pygame.quit()
          quit()
      # 获取环境渲染帧
      frame = env.render()
      if frame is not None:
        # 将numpy数组转换为pygame surface
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        # 创建显示训练信息的文字surface
        text_surface = font.render(
          f"Episode: {e+1} Score: {time}", True, (255, 0, 0)
        )
        # 缩放surface以匹配窗口尺寸
        scaled_surface = pygame.transform.scale(surface, (800, 600))
        # 将文字叠加到缩放后的surface上
        scaled_surface.blit(text_surface, (10, 10))
        # 将最终surface绘制到屏幕上
        screen.blit(scaled_surface, (0, 0))
        # 更新显示
        pygame.display.flip()
      # 智能体选择动作
      action = agent.act(state)
      # 执行动作，获取下一个状态、奖励等信息
      next_state, reward, terminated, truncated, _ = env.step(action)
      # 判断是否结束
      done = terminated or truncated
      # 如果结束，设置负奖励
      reward = reward if not done else -10
      # 处理下一个状态
      next_state = np.array(next_state, dtype=np.float32).flatten()
      next_state = np.reshape(next_state, [1, state_size])
      # 将经验存储到回hfill缓冲区
      agent.remember(state, action, reward, next_state, done)
      # 更新当前状态
      state = next_state
      # 记录当前得分
      # 如果结束，打印信息并跳出本轮循环
      if done:
        print(
          f"Episode: {e + 1}/{EPISODES}, Score: {time}, Epsilon: {agent.epsilon:.2f}"
        )
        break
      scores_list.append(time)
    # 如果回hfill缓冲区中有足够数据，进行训练
    if len(agent.memory) > batch_size:
      agent.replay(batch_size)
      #保存模型
      if e % 100 == 0:
        torch.save(agent.model.state_dict(), f"model_{e}.pth")
      
  # 绘制分数曲线
  plt.plot(scores_list)
  plt.xlabel('Episode')
  plt.ylabel('Score')
  plt.title('Training Scores')
  plt.show()

 
  # 关闭环境
  env.close()
  # 退出pygame
  pygame.quit()