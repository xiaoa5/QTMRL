"""可视化模块"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional


def plot_portfolio_value(
    portfolio_values: np.ndarray,
    dates: Optional[np.ndarray] = None,
    title: str = "组合净值曲线",
    save_path: Optional[str] = None,
):
    """绘制组合净值曲线

    Args:
        portfolio_values: 组合价值序列
        dates: 日期序列
        title: 图表标题
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 6))

    if dates is not None:
        plt.plot(dates, portfolio_values, linewidth=2)
        plt.xticks(rotation=45)
    else:
        plt.plot(portfolio_values, linewidth=2)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("日期" if dates is not None else "交易日", fontsize=12)
    plt.ylabel("组合价值 ($)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图表已保存到 {save_path}")

    plt.show()


def plot_drawdown(
    portfolio_values: np.ndarray,
    dates: Optional[np.ndarray] = None,
    title: str = "回撤曲线",
    save_path: Optional[str] = None,
):
    """绘制回撤曲线

    Args:
        portfolio_values: 组合价值序列
        dates: 日期序列
        title: 图表标题
        save_path: 保存路径
    """
    # 计算回撤
    cummax = np.maximum.accumulate(portfolio_values)
    drawdowns = (portfolio_values - cummax) / cummax

    plt.figure(figsize=(12, 6))

    if dates is not None:
        plt.fill_between(dates, drawdowns * 100, 0, alpha=0.3, color="red")
        plt.plot(dates, drawdowns * 100, color="red", linewidth=2)
        plt.xticks(rotation=45)
    else:
        plt.fill_between(range(len(drawdowns)), drawdowns * 100, 0, alpha=0.3, color="red")
        plt.plot(drawdowns * 100, color="red", linewidth=2)

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("日期" if dates is not None else "交易日", fontsize=12)
    plt.ylabel("回撤 (%)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图表已保存到 {save_path}")

    plt.show()


def plot_returns_distribution(
    returns: np.ndarray,
    title: str = "收益率分布",
    save_path: Optional[str] = None,
):
    """绘制收益率分布

    Args:
        returns: 收益率序列
        title: 图表标题
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 直方图
    axes[0].hist(returns * 100, bins=50, alpha=0.7, color="blue", edgecolor="black")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[0].set_title("收益率直方图", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("收益率 (%)", fontsize=10)
    axes[0].set_ylabel("频数", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 时间序列
    axes[1].plot(returns * 100, linewidth=1, alpha=0.7)
    axes[1].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[1].set_title("收益率时间序列", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("交易日", fontsize=10)
    axes[1].set_ylabel("收益率 (%)", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图表已保存到 {save_path}")

    plt.show()


def plot_action_distribution(
    actions_history: np.ndarray,
    asset_names: list,
    title: str = "动作分布",
    save_path: Optional[str] = None,
):
    """绘制动作分布

    Args:
        actions_history: [T, N] 动作历史
        asset_names: 资产名称列表
        title: 图表标题
        save_path: 保存路径
    """
    action_names = ["SELL", "HOLD", "BUY"]
    n_assets = len(asset_names)

    fig, axes = plt.subplots(1, n_assets, figsize=(4 * n_assets, 5))

    if n_assets == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        asset_actions = actions_history[:, i]
        action_counts = [
            np.sum(asset_actions == 0),
            np.sum(asset_actions == 1),
            np.sum(asset_actions == 2),
        ]

        ax.bar(action_names, action_counts, color=["red", "gray", "green"], alpha=0.7)
        ax.set_title(f"{asset_names[i]}", fontsize=10, fontweight="bold")
        ax.set_ylabel("次数", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图表已保存到 {save_path}")

    plt.show()


def plot_training_curves(
    metrics_history: dict,
    save_path: Optional[str] = None,
):
    """绘制训练曲线

    Args:
        metrics_history: 指标历史字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    if "actor_loss" in metrics_history:
        axes[0, 0].plot(metrics_history["actor_loss"], label="Actor Loss", alpha=0.7)
    if "value_loss" in metrics_history:
        axes[0, 0].plot(metrics_history["value_loss"], label="Value Loss", alpha=0.7)
    axes[0, 0].set_title("损失曲线", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("步数", fontsize=10)
    axes[0, 0].set_ylabel("损失", fontsize=10)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Entropy
    if "entropy" in metrics_history:
        axes[0, 1].plot(metrics_history["entropy"], color="purple", alpha=0.7)
        axes[0, 1].set_title("熵曲线", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("步数", fontsize=10)
        axes[0, 1].set_ylabel("熵", fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)

    # Reward
    if "avg_reward" in metrics_history:
        axes[1, 0].plot(metrics_history["avg_reward"], color="green", alpha=0.7)
        axes[1, 0].set_title("平均奖励", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("步数", fontsize=10)
        axes[1, 0].set_ylabel("奖励", fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)

    # Portfolio Value
    if "portfolio_value" in metrics_history:
        axes[1, 1].plot(metrics_history["portfolio_value"], color="blue", alpha=0.7)
        axes[1, 1].set_title("组合价值", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("步数", fontsize=10)
        axes[1, 1].set_ylabel("价值 ($)", fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图表已保存到 {save_path}")

    plt.show()
