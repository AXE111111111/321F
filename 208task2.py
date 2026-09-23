# ==========================================
# 园丁施肥问题 - 马尔可夫决策过程 (MDP) 求解
# ==========================================

# 1. 业务参数设置 (对照课件矩阵)
# 为了方便列表索引，我们在代码里把状态和动作映射为 0, 1, 2
# 状态: 0=好(Good), 1=中(Fair), 2=差(Poor)
# 动作: 0=不施肥(对应课件a=1), 1=施肥(对应课件a=2)

# 转移概率矩阵 P[action][current_state][next_state]
P = [
    # 动作 0: 不施肥 (课件 P1矩阵)
    [
        [0.2, 0.5, 0.3],  # 如果当前是'好'
        [0.0, 0.5, 0.5],  # 如果当前是'中'
        [0.0, 0.0, 1.0]  # 如果当前是'差'
    ],
    # 动作 1: 施肥 (课件 P2矩阵)
    [
        [0.3, 0.6, 0.1],
        [0.1, 0.6, 0.3],
        [0.05, 0.4, 0.55]
    ]
]

# 奖励矩阵 R[action][current_state][next_state]
R = [
    # 动作 0: 不施肥 (课件 R1矩阵)
    [
        [7, 6, 3],
        [0, 5, 1],
        [0, 0, -1]
    ],
    # 动作 1: 施肥 (课件 R2矩阵)
    [
        [6, 5, -1],
        [7, 4, 0],
        [6, 3, -2]
    ]
]

N_years = 3  # 规划期为 3 年
num_states = 3  # 3 种土壤状态
num_actions = 2  # 2 种动作

# 2. 初始化 DP 数组 (我们的小本本)
# f[n][i] 表示: 第 n 年，如果土壤状态为 i，到最后能拿到的【最大预期总收益】
# 因为规划期只有3年，所以第4年(索引3)的未来收益全部初始化为 0 [cite: 799]
f = [[0.0] * num_states for _ in range(N_years + 1)]

# 用来记录每一步应该做啥动作的说明书
optimal_policy = [[-1] * num_states for _ in range(N_years)]

# 3. 核心计算：从最后一年开始倒推 (Backward Recursion) [cite: 808]
# n 从 2 倒数到 0 (代表课件里的第3年，第2年，第1年)
for n in range(N_years - 1, -1, -1):

    # 遍历每一种可能的土壤状态
    for i in range(num_states):
        max_reward = -float('inf')  # 初始最大收益设为极小值 [cite: 810]
        best_action = -1

        # 针对当前状态，挨个尝试所有动作，看哪个划算 [cite: 811]
        for a in range(num_actions):
            expected_reward = 0

            # 平行宇宙分裂：遍历下一个阶段所有可能发生的状态 j
            for j in range(num_states):
                # 【灵魂公式】: 概率 * (当期立刻拿到的钱 + 未来小本本里的保底钱) [cite: 813]
                expected_reward += P[a][i][j] * (R[a][i][j] + f[n + 1][j])

            # 打擂台，找出能带来最大收益的动作 [cite: 814, 815]
            if expected_reward > max_reward:
                max_reward = expected_reward
                best_action = a

        # 把算出来的最高收益和对应的绝佳动作，写进小本本里 [cite: 816, 817]
        f[n][i] = round(max_reward, 2)
        optimal_policy[n][i] = best_action

# 4. 打印最终的“武功秘籍”
state_names = ["好", "中", "差"]
action_names = ["不施肥", "施肥"]

print("=== MDP 动态规划最优策略 ===")
for n in range(N_years):
    print(f"\n第 {n + 1} 年:")
    for i in range(num_states):
        print(f"  土壤状态 '{state_names[i]}': "
              f"动作 -> 【{action_names[optimal_policy[n][i]]}】, "
              f"预期未来总收益 -> {f[n][i]}")