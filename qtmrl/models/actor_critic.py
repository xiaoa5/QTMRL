"""Actor-Critic 模型"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import TimeCNNEncoder, TransformerEncoder


class MultiHeadActor(nn.Module):
    """多头Actor网络（每个资产一个头）"""

    def __init__(self, encoder: nn.Module, d_model: int, n_assets: int, n_actions: int = 3):
        """初始化Actor

        Args:
            encoder: 特征编码器
            d_model: 模型维度
            n_assets: 资产数量
            n_actions: 每个资产的动作数量（默认3: SELL/HOLD/BUY）
        """
        super().__init__()

        self.encoder = encoder
        self.n_assets = n_assets
        self.n_actions = n_actions

        # 每个资产的动作头
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, n_actions),
            )
            for _ in range(n_assets)
        ])

    def forward(self, features, positions, cash):
        """前向传播

        Args:
            features: [B, W, N, F] 特征张量
            positions: [B, N] 持仓比例
            cash: [B, 1] 现金比例

        Returns:
            logits: [B, N, 3] 动作logits
            probs: [B, N, 3] 动作概率分布
        """
        # 编码
        encodings = self.encoder(features, positions, cash)  # [B, N, d_model]

        B, N, _ = encodings.shape
        assert N == self.n_assets

        # 对每个资产计算动作分布
        logits_list = []
        for i in range(N):
            asset_enc = encodings[:, i, :]  # [B, d_model]
            logits = self.action_heads[i](asset_enc)  # [B, n_actions]
            logits_list.append(logits)

        # 堆叠 [B, N, n_actions]
        logits = torch.stack(logits_list, dim=1)

        # 计算概率
        probs = F.softmax(logits, dim=-1)

        return logits, probs

    def sample_action(self, features, positions, cash, deterministic=False):
        """采样动作

        Args:
            features: [B, W, N, F]
            positions: [B, N]
            cash: [B, 1]
            deterministic: 是否使用确定性策略（argmax）

        Returns:
            actions: [B, N] 动作索引
            log_probs: [B, N] 每个动作的log概率
        """
        logits, probs = self.forward(features, positions, cash)

        if deterministic:
            # 确定性策略：选择概率最大的动作
            actions = torch.argmax(probs, dim=-1)  # [B, N]
            log_probs = torch.log(probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1) + 1e-8)
        else:
            # 随机采样
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()  # [B, N]
            log_probs = dist.log_prob(actions)  # [B, N]

        return actions, log_probs

    def evaluate_actions(self, features, positions, cash, actions):
        """评估给定动作的log概率和熵

        Args:
            features: [B, W, N, F]
            positions: [B, N]
            cash: [B, 1]
            actions: [B, N] 动作索引

        Returns:
            log_probs: [B] 总log概率（所有资产的log概率之和）
            entropy: [B] 总熵
        """
        logits, probs = self.forward(features, positions, cash)

        # 计算每个资产的log概率
        dist = torch.distributions.Categorical(probs)
        log_probs_per_asset = dist.log_prob(actions)  # [B, N]

        # 总log概率（factorized policy）
        log_probs = log_probs_per_asset.sum(dim=-1)  # [B]

        # 计算熵
        entropy_per_asset = dist.entropy()  # [B, N]
        entropy = entropy_per_asset.mean(dim=-1)  # [B]

        return log_probs, entropy


class Critic(nn.Module):
    """Critic网络（价值函数）"""

    def __init__(self, encoder: nn.Module, d_model: int, n_assets: int):
        """初始化Critic

        Args:
            encoder: 特征编码器（可以与Actor共享）
            d_model: 模型维度
            n_assets: 资产数量
        """
        super().__init__()

        self.encoder = encoder
        self.n_assets = n_assets

        # 全局聚合层
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, features, positions, cash):
        """前向传播

        Args:
            features: [B, W, N, F] 特征张量
            positions: [B, N] 持仓比例
            cash: [B, 1] 现金比例

        Returns:
            value: [B] 状态价值
        """
        # 编码
        encodings = self.encoder(features, positions, cash)  # [B, N, d_model]

        # 全局聚合（跨资产）
        # [B, N, d_model] -> [B, d_model, N] -> [B, d_model, 1] -> [B, d_model]
        global_enc = encodings.permute(0, 2, 1)  # [B, d_model, N]
        global_enc = self.global_pool(global_enc).squeeze(-1)  # [B, d_model]

        # 价值估计
        value = self.value_head(global_enc).squeeze(-1)  # [B]

        return value


def create_models(config, n_assets, n_features):
    """根据配置创建Actor和Critic模型

    Args:
        config: 配置对象
        n_assets: 资产数量
        n_features: 特征数量

    Returns:
        (actor, critic) 模型元组
    """
    d_model = config.model.d_model
    n_layers = config.model.n_layers
    dropout = config.model.dropout
    encoder_type = config.model.encoder

    # 创建两个独立的编码器（也可以共享）
    if encoder_type == "TimeCNN":
        actor_encoder = TimeCNNEncoder(
            n_assets=n_assets,
            n_features=n_features,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )
        critic_encoder = TimeCNNEncoder(
            n_assets=n_assets,
            n_features=n_features,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )
    elif encoder_type == "Transformer":
        actor_encoder = TransformerEncoder(
            n_assets=n_assets,
            n_features=n_features,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=4,
            dropout=dropout,
        )
        critic_encoder = TransformerEncoder(
            n_assets=n_assets,
            n_features=n_features,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=4,
            dropout=dropout,
        )
    else:
        raise ValueError(f"未知的编码器类型: {encoder_type}")

    # 创建Actor和Critic
    actor = MultiHeadActor(actor_encoder, d_model, n_assets)
    critic = Critic(critic_encoder, d_model, n_assets)

    return actor, critic
