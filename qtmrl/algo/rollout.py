"""Rollout缓冲区"""
import torch
import numpy as np
from typing import List, Dict


class RolloutBuffer:
    """存储rollout数据的缓冲区"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置缓冲区"""
        self.features = []
        self.positions = []
        self.cash = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        cash: torch.Tensor,
        actions: torch.Tensor,
        rewards: float,
        values: torch.Tensor,
        log_probs: torch.Tensor,
        dones: bool,
    ):
        """添加一步数据

        Args:
            features: [W, N, F] 特征
            positions: [N] 持仓
            cash: [1] 现金
            actions: [N] 动作
            rewards: 标量奖励
            values: [1] 价值估计
            log_probs: [N] log概率
            dones: 是否结束
        """
        self.features.append(features)
        self.positions.append(positions)
        self.cash.append(cash)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.values.append(values)
        self.log_probs.append(log_probs)
        self.dones.append(dones)

    def get(self) -> Dict[str, torch.Tensor]:
        """获取批量数据

        Returns:
            数据字典
        """
        # 转换为tensor
        features = torch.stack(self.features)  # [T, W, N, F]
        positions = torch.stack(self.positions)  # [T, N]
        cash = torch.stack(self.cash)  # [T, 1]
        actions = torch.stack(self.actions)  # [T, N]
        rewards = torch.tensor(self.rewards, dtype=torch.float32)  # [T]
        values = torch.stack(self.values)  # [T]
        log_probs = torch.stack(self.log_probs)  # [T, N]
        dones = torch.tensor(self.dones, dtype=torch.bool)  # [T]

        return {
            "features": features,
            "positions": positions,
            "cash": cash,
            "actions": actions,
            "rewards": rewards,
            "values": values,
            "log_probs": log_probs,
            "dones": dones,
        }

    def __len__(self):
        return len(self.rewards)


def compute_returns_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.96,
) -> tuple:
    """计算回报和优势

    Args:
        rewards: [T] 奖励序列
        values: [T] 价值估计序列
        last_value: 标量，最后状态的价值估计
        dones: [T] 是否结束
        gamma: 折扣因子

    Returns:
        (returns, advantages)
        - returns: [T] 回报
        - advantages: [T] 优势
    """
    T = len(rewards)
    returns = torch.zeros(T, dtype=torch.float32)
    advantages = torch.zeros(T, dtype=torch.float32)

    # 从后向前计算
    next_value = last_value
    next_return = last_value

    for t in reversed(range(T)):
        if dones[t]:
            next_value = 0.0
            next_return = 0.0

        # 计算回报 (bootstrapped return)
        returns[t] = rewards[t] + gamma * next_return

        # 计算优势 (TD error)
        advantages[t] = rewards[t] + gamma * next_value - values[t]

        next_value = values[t]
        next_return = returns[t]

    return returns, advantages
