"""A2C训练器"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict
from .rollout import RolloutBuffer, compute_returns_advantages


class A2CTrainer:
    """A2C算法训练器"""

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        lr_actor: float = 1e-5,
        lr_critic: float = 1e-5,
        gamma: float = 0.96,
        entropy_coef: float = 0.05,
        value_coef: float = 1.0,
        grad_clip: float = 1.0,
        device: str = "cpu",
    ):
        """初始化A2C训练器

        Args:
            actor: Actor模型
            critic: Critic模型
            lr_actor: Actor学习率
            lr_critic: Critic学习率
            gamma: 折扣因子
            entropy_coef: 熵系数
            value_coef: 价值损失系数
            grad_clip: 梯度裁剪阈值
            device: 设备
        """
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.grad_clip = grad_clip
        self.device = device

        # 优化器
        self.actor_optimizer = optim.Adam(actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(critic.parameters(), lr=lr_critic)

    def collect_rollout(
        self, env, rollout_steps: int, buffer: RolloutBuffer
    ) -> Dict:
        """收集rollout数据

        Args:
            env: 交易环境
            rollout_steps: rollout步数
            buffer: rollout缓冲区

        Returns:
            统计信息字典
        """
        buffer.reset()
        total_reward = 0.0
        steps = 0

        for _ in range(rollout_steps):
            # 获取当前状态
            state = env._get_state()

            # 转换为tensor
            features = torch.from_numpy(state["features"]).unsqueeze(0).to(self.device)
            positions = torch.from_numpy(state["positions"]).unsqueeze(0).to(self.device)
            cash = torch.from_numpy(state["cash"]).unsqueeze(0).to(self.device)

            # 采样动作
            with torch.no_grad():
                actions, log_probs = self.actor.sample_action(
                    features, positions, cash, deterministic=False
                )
                values = self.critic(features, positions, cash)

            # 执行动作
            actions_np = actions.cpu().numpy()[0]
            next_state, reward, done, info = env.step(actions_np)

            # 存储
            buffer.add(
                features=features.squeeze(0).cpu(),
                positions=positions.squeeze(0).cpu(),
                cash=cash.squeeze(0).cpu(),
                actions=actions.squeeze(0).cpu(),
                rewards=reward,
                values=values.cpu(),
                log_probs=log_probs.squeeze(0).cpu(),
                dones=done,
            )

            total_reward += reward
            steps += 1

            if done:
                break

        # 计算最后状态的价值（用于bootstrap）
        if done:
            last_value = torch.tensor(0.0, dtype=torch.float32)
        else:
            with torch.no_grad():
                next_features = torch.from_numpy(next_state["features"]).unsqueeze(0).to(self.device)
                next_positions = torch.from_numpy(next_state["positions"]).unsqueeze(0).to(self.device)
                next_cash = torch.from_numpy(next_state["cash"]).unsqueeze(0).to(self.device)
                last_value = self.critic(next_features, next_positions, next_cash).cpu()

        stats = {
            "avg_reward": total_reward / max(steps, 1),
            "total_reward": total_reward,
            "steps": steps,
            "last_value": last_value.item(),
        }

        return stats, last_value

    def update(self, buffer: RolloutBuffer, last_value: torch.Tensor) -> Dict:
        """更新Actor和Critic

        Args:
            buffer: rollout缓冲区
            last_value: 最后状态的价值估计

        Returns:
            损失统计字典
        """
        # 获取数据
        data = buffer.get()

        features = data["features"].to(self.device)
        positions = data["positions"].to(self.device)
        cash = data["cash"].to(self.device)
        actions = data["actions"].to(self.device)
        old_log_probs = data["log_probs"]
        rewards = data["rewards"]
        values = data["values"]
        dones = data["dones"]

        # 计算回报和优势
        returns, advantages = compute_returns_advantages(
            rewards, values, last_value, dones, self.gamma
        )

        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        # 标准化优势
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ========== 更新Actor ==========
        self.actor_optimizer.zero_grad()

        # 重新计算log概率和熵
        log_probs, entropy = self.actor.evaluate_actions(features, positions, cash, actions)

        # Actor损失（Policy Gradient + Entropy Bonus）
        actor_loss = -(log_probs * advantages.detach()).mean() - self.entropy_coef * entropy.mean()

        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        # ========== 更新Critic ==========
        self.critic_optimizer.zero_grad()

        # 重新计算价值
        new_values = self.critic(features, positions, cash)

        # Critic损失（MSE）
        value_loss = nn.functional.mse_loss(new_values, returns.detach())

        critic_loss = self.value_coef * value_loss
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # 统计
        stats = {
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "advantages_mean": advantages.mean().item(),
            "returns_mean": returns.mean().item(),
        }

        return stats

    def train_step(self, env, rollout_steps: int) -> Dict:
        """执行一次训练步骤

        Args:
            env: 交易环境
            rollout_steps: rollout步数

        Returns:
            统计信息字典
        """
        # 设置为训练模式
        self.actor.train()
        self.critic.train()

        # 收集rollout
        buffer = RolloutBuffer()
        rollout_stats, last_value = self.collect_rollout(env, rollout_steps, buffer)

        # 更新模型
        update_stats = self.update(buffer, last_value)

        # 合并统计
        stats = {**rollout_stats, **update_stats}

        return stats

    def save(self, path: str):
        """保存模型

        Args:
            path: 保存路径
        """
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """加载模型

        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
