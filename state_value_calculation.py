# 强化学习状态价值计算示例
import numpy as np

# 1. 定义网格世界环境参数
GRID_SIZE = 4  # 4x4网格世界
TERMINAL_STATES = [(0,0), (3,3)]  # 终止状态坐标
ACTIONS = ['up', 'down', 'left', 'right']  # 可能的动作
GAMMA = 0.9  # 折扣因子
THETA = 1e-4  # 收敛阈值

# 2. 初始化状态价值函数
V = np.zeros((GRID_SIZE, GRID_SIZE))

# 3. 定义状态转移函数
def get_next_state(s, a):
    """根据当前状态和动作返回下一个状态"""
    i, j = s
    if s in TERMINAL_STATES:
        return s  # 终止状态不再转移
    
    if a == 'up':
        next_i = max(i-1, 0)
        next_j = j
    elif a == 'down':
        next_i = min(i+1, GRID_SIZE-1)
        next_j = j
    elif a == 'left':
        next_i = i
        next_j = max(j-1, 0)
    elif a == 'right':
        next_i = i
        next_j = min(j+1, GRID_SIZE-1)
        
    return (next_i, next_j)

# 4. 策略评估迭代
iteration = 0
while True:
    delta = 0
    new_V = np.zeros_like(V)
    
    # 遍历所有状态
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            s = (i, j)
            if s in TERMINAL_STATES:
                new_V[i,j] = 0  # 终止状态价值保持0
                continue
                
            # 计算当前状态的价值（假设均匀随机策略）
            total_value = 0
            for a in ACTIONS:
                next_s = get_next_state(s, a)
                # 即时奖励：每步-1，到达终止状态时结束
                reward = 1 if next_s not in TERMINAL_STATES else 0
                # 累加各动作的价值（均匀概率） 这里的意思是，每一步都有0.25的概率转移到下一个状态
                total_value += 0.25 * (reward + GAMMA * V[next_s])
                
            new_V[i,j] = total_value
            delta = max(delta, abs(new_V[i,j] - V[i,j]))
    
    # 更新价值函数并检查收敛
    V = new_V.copy()
    iteration += 1
    print(f"Iteration {iteration}:")
    print(np.round(V, 2))
    
    if delta < THETA:
        break

# 5. 输出最终结果
print("\nFinal state values:")
print(np.round(V, 2))
