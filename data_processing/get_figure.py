import data_processing.config as config

import matplotlib.pyplot as plt
import os

import sys
sys.path.append("./")

def get_figure(goal_list, label_list, title, ylabel):
    # 设置保存路径（可自定义）
    save_path = config.figure_path + title + ".png"

    # 创建文件夹（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 绘图
    for a, b in zip(goal_list, label_list):
        plt.plot(a, label=b, marker='o', linestyle='-')
    plt.ylabel(ylabel)
    plt.xlabel("Iteration")
    
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 可选：不显示图形，仅保存
    plt.close()

    print(f"✅ 已保存折线图到: {save_path}")
    
def get_figure_one(a, a_label, title):
    # 设置保存路径（可自定义）
    save_path = config.figure_path + title + ".png"

    # 创建文件夹（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 绘图
    plt.plot(a, marker='o', linestyle='-')
    plt.ylabel(a_label)
    plt.xlabel("Iteration")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # 可选：不显示图形，仅保存
    plt.close()

    print(f"✅ 已保存折线图到: {save_path}")
