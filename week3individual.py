import numpy as np
import matplotlib.pyplot as plt

# --- 物理与电路参数 ---
Vs = 5.0  # V
R1 = 100.0  # Ohm
R2 = 100.0  # Ohm
C1 = 10e-6  # F (10 uF)
C2 = 10e-6  # F (10 uF)
Is = 1e-12  # A
VT = 0.02585  # V


# --- 右端项 f(y) 与 雅可比矩阵 J_f ---
def f(y):
    v1, v2 = y
    dv1_dt = ((Vs - v1) / R1 - (v1 - v2) / R2) / C1

    # 限制指数项过大导致溢出
    exp_term = np.exp(np.clip(v2 / VT, -100, 100))
    dv2_dt = ((v1 - v2) / R2 - Is * (exp_term - 1)) / C2
    return np.array([dv1_dt, dv2_dt])


def J_f(y):
    v1, v2 = y
    df1_dv1 = (-1 / R1 - 1 / R2) / C1
    df1_dv2 = (1 / R2) / C1
    df2_dv1 = (1 / R2) / C2

    # 限制指数项过大导致溢出
    exp_term = np.exp(np.clip(v2 / VT, -100, 100))
    df2_dv2 = (-1 / R2 - (Is / VT) * exp_term) / C2
    return np.array([[df1_dv1, df1_dv2],
                     [df2_dv1, df2_dv2]])


# --- 牛顿迭代法求解第一步 (修正的 Backtracking 线搜索) ---
def solve_first_step(y_n, h, damping=False):
    y = np.copy(y_n)
    v2_history = [y[1]]

    for k in range(25):
        # 1. 计算残差 F 和 雅可比 J_F
        F = y - y_n - h * f(y)
        J_F = np.eye(2) - h * J_f(y)

        # 2. 求解增量 delta
        delta = np.linalg.solve(J_F, -F)

        # 3. 阻尼策略 (Backtracking line search)
        alpha = 1.0
        if damping:
            F_norm = np.linalg.norm(F)
            # 如果走完整步 alpha=1 会导致残差变大，就将步长减半
            while alpha > 1e-6:
                y_try = y + alpha * delta
                F_try = y_try - y_n - h * f(y_try)
                if np.linalg.norm(F_try) <= F_norm:
                    break  # 残差没有变大，接受当前步长
                alpha *= 0.5

        # 4. 更新状态
        y = y + alpha * delta
        v2_history.append(y[1])

        # 5. 收敛判定
        if np.linalg.norm(delta) < 1e-6:
            break

    return v2_history


# --- 运行模拟 ---
y0 = np.array([0.0, 0.0])  # 初始值
h = 1e-3  # 步长 1 ms

# 1. 改进前：全牛顿法 (Full Newton)
v2_full = solve_first_step(y0, h, damping=False)

# 2. 改进后：阻尼牛顿法 (Damped Newton)
v2_damped = solve_first_step(y0, h, damping=True)

# --- 绘图 ---
# 1. 保存改进前图像 (Before Improvement)
fig_before, ax1 = plt.subplots(figsize=(6, 4), dpi=150)
ax1.plot(v2_full, marker='o', color='#444444', label=f'Full Newton ({len(v2_full) - 1} iter)')
ax1.set_title("Before Improvement: First step, $h=1$ ms (Full Newton)")
ax1.set_xlabel("Newton iteration $k$")
ax1.set_ylabel("$v_2^{(k)}$ [V]")
ax1.axhline(0.586, color='gray', linestyle='--')
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend()
plt.tight_layout()
fig_before.savefig("figure_before.png")
plt.close(fig_before)

# 2. 保存改进后图像 (After Improvement)
fig_after, ax2 = plt.subplots(figsize=(6, 4), dpi=150)
ax2.plot(v2_damped, marker='s', color='#990033', label=f'Damped Newton ({len(v2_damped) - 1} iter)')
ax2.set_title("After Improvement: First step, $h=1$ ms (Damped Newton)")
ax2.set_xlabel("Newton iteration $k$")
ax2.set_ylabel("$v_2^{(k)}$ [V]")
ax2.axhline(0.586, color='gray', linestyle='--')
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend()
plt.tight_layout()
fig_after.savefig("figure_after.png")
plt.close(fig_after)

print("图片生成完毕：figure_before.png 和 figure_after.png")
