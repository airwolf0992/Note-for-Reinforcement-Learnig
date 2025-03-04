# 强化学习动作价值函数计算示例
import numpy as np

# 1. 定义环境参数
states = ['s0', 's1']  # 状态集合
actions = ['a0', 'a1']  # 动作集合

# 状态转移矩阵（确定性环境）
transitions = {
    's0': {'a0': 's1', 'a1': 's0'},
    's1': {'a0': 's0', 'a1': 's1'}
}

# 即时奖励矩阵
rewards = {
    's0': {'a0': 5, 'a1': -1},
    's1': {'a0': -2, 'a1': 2}
}

# 2. 初始化参数
gamma = 0.9  # 折扣因子
n_iterations = 100  # 迭代次数

# 3. 初始化Q表（动作价值函数）
Q = {s: {a: 0.0 for a in actions} for s in states}

# 4. 贝尔曼方程迭代更新
for _ in range(n_iterations):
    new_Q = Q.copy()
    for s in states:
        for a in actions:
            # 获取下一状态和即时奖励
            next_s = transitions[s][a]
            reward = rewards[s][a]
            
            # 计算最大下一状态价值（假设策略为贪婪策略）
            max_next_q = max(Q[next_s].values())
            
            # 更新Q值：Q(s,a) = r + γ * max_a' Q(s',a')
            new_Q[s][a] = reward + gamma * max_next_q
    
    Q = new_Q

# 5. 打印最终Q值表
print("迭代次数:", n_iterations)
print("折扣因子:", gamma)
print("\n最终动作价值函数 Q(s,a):")
for s in states:
    for a in actions:
        print(f"Q({s}, {a}) = {Q[s][a]:.2f}")
    print()
