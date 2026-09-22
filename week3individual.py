import numpy as np
import matplotlib.pyplot as plt

# 构造与原图横坐标一致的均匀步长网格 h
h_vals = np.array([6.25e-6, 1.25e-5, 2.5e-5, 5e-5, 1e-4])

# 构造与原图误差数据极度吻合的数据
err_explicit_euler = 6000 * h_vals ** 1.027
err_implicit_euler = 4000 * h_vals ** 0.973
err_rk4 = 1.5e11 * h_vals ** 4.002


def plot_convergence(add_reference_lines=False, filename="plot.png"):
    # 采用类似原图的白色网格主题
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor='white')

    # 画实际数据线 (实线带圆点)
    ax.loglog(h_vals, err_explicit_euler, marker='o', label='Explicit Euler: p=1.027', color='#377eb8')
    ax.loglog(h_vals, err_rk4, marker='o', label='RK4: p=4.002', color='#4daf4a')
    ax.loglog(h_vals, err_implicit_euler, marker='o', label='Implicit Euler: p=0.973', color='#d95f02')

    # 改进后的优化：添加理论参考虚线
    if add_reference_lines:
        # Explicit Euler 理论一阶 O(h) 虚线
        ref_euler = err_explicit_euler[-1] * (h_vals / h_vals[-1]) ** 1.0
        ax.loglog(h_vals, ref_euler, linestyle='--', color='#92c5de', zorder=0)

        # RK4 理论四阶 O(h^4) 虚线
        ref_rk4 = err_rk4[-1] * (h_vals / h_vals[-1]) ** 4.0
        ax.loglog(h_vals, ref_rk4, linestyle='--', color='#a6dba0', zorder=0)

        # Implicit Euler 理论一阶 O(h) 虚线
        ref_implicit = err_implicit_euler[-1] * (h_vals / h_vals[-1]) ** 1.0
        ax.loglog(h_vals, ref_implicit, linestyle='--', color='#f4a582', zorder=0)

    # 设置图表样式以匹配原图
    ax.set_title("PMSM: convergence against matrix-exponential reference", pad=15, fontsize=12)
    ax.set_xlabel("Uniform step h (s)")
    ax.set_ylabel("Max grid/component error (A)")

    # 开启网格并去除非必要边框
    ax.grid(True, which="major", linestyle='-', color='#e0e0e0', alpha=0.7)
    ax.grid(True, which="minor", linestyle='-', color='#f0f0f0', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 纵坐标范围设置
    ax.set_ylim([1e-10, 5])

    # 图例
    ax.legend(loc='lower right', frameon=True, edgecolor='lightgray')

    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


# 1. 生成改进前图片 (纯数据线)
plot_convergence(add_reference_lines=False, filename="pmsm_before.png")

# 2. 生成改进后图片 (增加了理论参考虚线)
plot_convergence(add_reference_lines=True, filename="pmsm_after.png")

print("图片生成完毕：pmsm_before.png 和 pmsm_after.png")
