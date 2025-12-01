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
        kernel_size: int = 3,
        window_size: int = None,  # Added window_size argument
    ):
        """初始化编码器

        Args:
            n_assets: 资产数量
            n_features: 每个资产的特征数量
            d_model: 模型维度
            n_layers: 卷积层数
            dropout: Dropout比例
            kernel_size: 卷积核大小（默认3，会根据输入自动调整）
            window_size: 窗口大小（可选，如果提供则预初始化层）
        """
        super().__init__()

        self.n_assets = n_assets
        self.n_features = n_features
        self.d_model = d_model
        self.base_kernel_size = kernel_size
        self.n_layers = n_layers
        self.dropout = dropout

        # 位置和现金信息融合
        self.pos_embed = nn.Linear(1, d_model // 4)
        self.cash_embed = nn.Linear(1, d_model // 4)
        
        # Initialize embedding layers
        nn.init.xavier_uniform_(self.pos_embed.weight)
        nn.init.zeros_(self.pos_embed.bias)
        nn.init.xavier_uniform_(self.cash_embed.weight)
        nn.init.zeros_(self.cash_embed.bias)
        
        # Note: Conv layers will be created dynamically in forward pass
        # to handle variable window sizes during validation
        self.conv_layers = None
        
        # Pre-initialize layers if window_size is provided (e.g. for evaluation)
        if window_size is not None:
            self._init_conv_layers(window_size)

    def _init_conv_layers(self, W):
        """Initialize conv layers based on window size"""
        # Dynamically create conv layers if not already created or if window size changed
        # Adjust kernel size based on window size to prevent errors
        kernel_size = min(self.base_kernel_size, W)
        if kernel_size < 1:
            kernel_size = 1
        
        # Use 'same' padding to maintain sequence length through all conv layers
        # For kernel_size=1, padding=0 is equivalent to 'same'
        if kernel_size == 1:
            padding = 0
        else:
            # Use 'same' padding mode to ensure output length = input length
            padding = 'same'
        
        layers = []
        
        # 第一层
        conv1 = nn.Conv1d(self.n_features, self.d_model, kernel_size=kernel_size, padding=padding)
        nn.init.xavier_uniform_(conv1.weight)
        nn.init.zeros_(conv1.bias)
        layers.append(conv1)
        layers.append(nn.ReLU())
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        
        # 中间层
        for _ in range(self.n_layers - 1):
            conv_layer = nn.Conv1d(self.d_model, self.d_model, kernel_size=kernel_size, padding=padding)
            nn.init.xavier_uniform_(conv_layer.weight)
            nn.init.zeros_(conv_layer.bias)
            layers.append(conv_layer)
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
        
        self.conv_layers = nn.Sequential(*layers)
        self._last_kernel_size = kernel_size

    def forward(self, features, positions, cash):
        """前向传播

        Args:
            features: [B, W, N, F] 特征张量
            positions: [B, N] 持仓比例
            cash: [B, 1] 现金比例

        Returns:
            [B, N, d_model] 编码后的资产表示
        """
        B, W, N, n_feat = features.shape
        assert N == self.n_assets, f"Expected {self.n_assets} assets, got {N}"
        assert n_feat == self.n_features, f"Expected {self.n_features} features, got {n_feat}"

        # For very small windows (W < 3), use linear projection to avoid conv issues
        # Conv1d with small inputs can have padding problems, especially with multiple layers
        if W < 3:
            # Use linear projection instead of convolution for very small windows
            if not hasattr(self, 'linear_proj'):
                self.linear_proj = nn.Linear(self.n_features, self.d_model).to(features.device)
                # Properly initialize the weights
                nn.init.xavier_uniform_(self.linear_proj.weight)
                nn.init.zeros_(self.linear_proj.bias)
            
            asset_encodings = []
            for i in range(N):
                # Average over the window dimension
                asset_feat = features[:, :, i, :]  # [B, W, F]
                asset_feat_avg = asset_feat.mean(dim=1)  # [B, F] - average over time
                
                # Check for NaN in input
                if torch.isnan(asset_feat_avg).any():
                    # Replace NaN with zeros
                    asset_feat_avg = torch.nan_to_num(asset_feat_avg, nan=0.0)
                
                encoded = self.linear_proj(asset_feat_avg)  # [B, d_model]
                encoded = F.relu(encoded)
                
                # Clamp to prevent extreme values
                encoded = torch.clamp(encoded, min=-10.0, max=10.0)
                
                asset_encodings.append(encoded)
            
            encodings = torch.stack(asset_encodings, dim=1)  # [B, N, d_model]
            return encodings

        # Dynamically create conv layers if not already created or if window size changed
        # Adjust kernel size based on window size to prevent errors
        kernel_size = min(self.base_kernel_size, W)
        if kernel_size < 1:
            kernel_size = 1
        
        # Use 'same' padding to maintain sequence length through all conv layers
        # For kernel_size=1, padding=0 is equivalent to 'same'
        if kernel_size == 1:
            padding = 0
        else:
            # Use 'same' padding mode to ensure output length = input length
            padding = 'same'
        
        # Build conv layers if needed
        if self.conv_layers is None or not hasattr(self, '_last_kernel_size') or self._last_kernel_size != kernel_size:
            layers = []
            
            # 第一层
            conv1 = nn.Conv1d(self.n_features, self.d_model, kernel_size=kernel_size, padding=padding)
            nn.init.xavier_uniform_(conv1.weight)
            nn.init.zeros_(conv1.bias)
            layers.append(conv1)
            layers.append(nn.ReLU())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            
            # 中间层
            for _ in range(self.n_layers - 1):
                conv_layer = nn.Conv1d(self.d_model, self.d_model, kernel_size=kernel_size, padding=padding)
                nn.init.xavier_uniform_(conv_layer.weight)
                nn.init.zeros_(conv_layer.bias)
                layers.append(conv_layer)
                layers.append(nn.ReLU())
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
            
            self.conv_layers = nn.Sequential(*layers)
            self._last_kernel_size = kernel_size
            
            # Move to same device as input
            self.conv_layers = self.conv_layers.to(features.device)
        
        # 全局平均池化
        pool = nn.AdaptiveAvgPool1d(1)

        # 对每个资产独立编码
        asset_encodings = []
        for i in range(N):
            # 提取单个资产的特征 [B, W, F]
            asset_feat = features[:, :, i, :]  # [B, W, F]

            # Check for NaN in input
            if torch.isnan(asset_feat).any():
                asset_feat = torch.nan_to_num(asset_feat, nan=0.0)

            # 转换为 [B, F, W] 用于Conv1d
            asset_feat = asset_feat.permute(0, 2, 1)  # [B, F, W]

            try:
                # 卷积编码
                encoded = self.conv_layers(asset_feat)  # [B, d_model, W']
                
                # Check for NaN after convolution
                if torch.isnan(encoded).any():
                    raise RuntimeError(f"NaN detected after convolution for asset {i}")
                
            except RuntimeError as e:
                # Provide detailed error message for debugging
                raise RuntimeError(
                    f"Conv1d failed with input shape {asset_feat.shape}, "
                    f"kernel_size={kernel_size}, padding={padding}, "
                    f"window_size={W}. Input stats: min={asset_feat.min():.4f}, "
                    f"max={asset_feat.max():.4f}, mean={asset_feat.mean():.4f}. "
                    f"Original error: {e}"
                )

            # 全局池化
            pooled = pool(encoded).squeeze(-1)  # [B, d_model]
            
            # Clamp to prevent extreme values
            pooled = torch.clamp(pooled, min=-10.0, max=10.0)

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
        B, W, N, n_feat = features.shape

        # 对每个资产独立编码
        asset_encodings = []
        for i in range(N):
            # 提取单个资产的特征 [B, W, n_feat]
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
