"""特征编码器模块"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeCNNEncoder(nn.Module):
    """基于1D卷积的时间序列编码器"""

    def __init__(
        self,
        n_assets: int,
        n_features: int,
        d_model: int = 128,
        n_layers: int = 3,
        dropout: float = 0.0,
    ):
        """初始化编码器

        Args:
            n_assets: 资产数量
            n_features: 每个资产的特征数量
            d_model: 模型维度
            n_layers: 卷积层数
            dropout: Dropout比例
        """
        super().__init__()

        self.n_assets = n_assets
        self.n_features = n_features
        self.d_model = d_model

        # 对每个资产独立编码
        # 输入: [B, W, F] -> 输出: [B, d_model]
        layers = []

        # 第一层
        layers.append(nn.Conv1d(n_features, d_model, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # 中间层
        for _ in range(n_layers - 1):
            layers.append(nn.Conv1d(d_model, d_model, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.conv_layers = nn.Sequential(*layers)

        # 全局平均池化
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 位置和现金信息融合
        self.pos_embed = nn.Linear(1, d_model // 4)
        self.cash_embed = nn.Linear(1, d_model // 4)

    def forward(self, features, positions, cash):
        """前向传播

        Args:
            features: [B, W, N, F] 特征张量
            positions: [B, N] 持仓比例
            cash: [B, 1] 现金比例

        Returns:
            [B, N, d_model] 编码后的资产表示
        """
        B, W, N, F = features.shape
        assert N == self.n_assets
        assert F == self.n_features

        # 对每个资产独立编码
        asset_encodings = []
        for i in range(N):
            # 提取单个资产的特征 [B, W, F]
            asset_feat = features[:, :, i, :]  # [B, W, F]

            # 转换为 [B, F, W] 用于Conv1d
            asset_feat = asset_feat.permute(0, 2, 1)  # [B, F, W]

            # 卷积编码
            encoded = self.conv_layers(asset_feat)  # [B, d_model, W]

            # 全局池化
            pooled = self.pool(encoded).squeeze(-1)  # [B, d_model]

            # 融合持仓信息
            pos_emb = self.pos_embed(positions[:, i:i+1])  # [B, d_model//4]
            pos_emb = F.relu(pos_emb)

            # 拼接
            # 为了保持维度一致，我们将pooled截断或者调整pos_emb维度
            # 这里简化处理：将位置嵌入加到编码上（广播）
            # 更好的做法是concat后再过一层MLP
            asset_encodings.append(pooled)

        # 堆叠所有资产 [B, N, d_model]
        encodings = torch.stack(asset_encodings, dim=1)

        return encodings


class TransformerEncoder(nn.Module):
    """基于Transformer的时间序列编码器"""

    def __init__(
        self,
        n_assets: int,
        n_features: int,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        """初始化Transformer编码器

        Args:
            n_assets: 资产数量
            n_features: 每个资产的特征数量
            d_model: 模型维度
            n_layers: Transformer层数
            n_heads: 注意力头数
            dropout: Dropout比例
        """
        super().__init__()

        self.n_assets = n_assets
        self.n_features = n_features
        self.d_model = d_model

        # 输入投影
        self.input_proj = nn.Linear(n_features, d_model)

        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))  # 最多100步

    def forward(self, features, positions, cash):
        """前向传播

        Args:
            features: [B, W, N, F] 特征张量
            positions: [B, N] 持仓比例
            cash: [B, 1] 现金比例

        Returns:
            [B, N, d_model] 编码后的资产表示
        """
        B, W, N, F = features.shape

        # 对每个资产独立编码
        asset_encodings = []
        for i in range(N):
            # 提取单个资产的特征 [B, W, F]
            asset_feat = features[:, :, i, :]

            # 投影到d_model维度
            x = self.input_proj(asset_feat)  # [B, W, d_model]

            # 添加位置编码
            x = x + self.pos_encoding[:, :W, :]

            # Transformer编码
            encoded = self.transformer(x)  # [B, W, d_model]

            # 取最后一个时间步
            last_hidden = encoded[:, -1, :]  # [B, d_model]

            asset_encodings.append(last_hidden)

        # 堆叠所有资产 [B, N, d_model]
        encodings = torch.stack(asset_encodings, dim=1)

        return encodings
