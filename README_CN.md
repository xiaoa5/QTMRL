# QTMRL - 基于 A2C 的多资产量化交易强化学习系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

QTMRL (Quantitative Trading with Multi-asset Reinforcement Learning) 是一个基于 **A2C (Advantage Actor-Critic)** 算法的多资产量化交易强化学习系统。该系统使用日频OHLCV数据和技术指标，通过factorized multi-head policy学习多资产交易策略，支持共享资金池和组合级奖励。

## 特性

- ✅ **完全可复现**: 固定随机种子，自动下载数据，一键运行
- 📊 **多资产交易**: 支持多只股票同时交易，共享资金池
- 🧠 **A2C算法**: 基于Advantage Actor-Critic的策略梯度方法
- 📈 **丰富的技术指标**: SMA, EMA, RSI, MACD, ATR, Bollinger Bands, Ichimoku, SuperTrend等
- 🔧 **灵活配置**: YAML配置文件，轻松修改参数
- 📉 **完整评估**: 收益率、夏普比率、波动率、最大回撤等指标
- 🎨 **可视化**: 净值曲线、回撤曲线、收益率分布、动作分布等
- 🚀 **Colab支持**: 适配Google Colab环境，支持GPU训练
- 📝 **Wandb集成**: 支持实验跟踪和可视化

## 快速开始

### 1. 环境要求

- Python 3.8+
- CUDA (可选，用于GPU加速)

### 2. 安装

#### 本地安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/QTMRL.git
cd QTMRL

# 安装依赖
pip install -r requirements.txt

# 或使用开发模式安装
pip install -e .
```

#### Google Colab安装

```python
# 在Colab notebook中运行
!git clone https://github.com/yourusername/QTMRL.git
%cd QTMRL
!pip install -r requirements.txt
```

### 3. 运行完整流程

#### 步骤 1: 数据预处理

下载股票数据并计算技术指标：

```bash
# 使用默认配置（16只股票，2014-2024）
python scripts/preprocess.py --config configs/default.yaml

# 或使用快速测试配置（4只股票，2022-2024）
python scripts/preprocess.py --config configs/quick_test.yaml
```

处理后的数据将保存在 `data/processed/` 目录。

#### 步骤 2: 训练模型

```bash
# 完整训练（1M步，约2-3小时）
python scripts/train.py --config configs/default.yaml

# 快速测试（50K步，约10-20分钟）
python scripts/train.py --config configs/quick_test.yaml
```

训练过程中会：
- 自动保存checkpoint
- 定期在验证集上评估
- 记录训练指标（loss, entropy, reward等）
- 支持Wandb可视化（可选）

#### 步骤 3: 评估模型

```bash
# 在测试集上评估
python scripts/evaluate.py \
    --config configs/default.yaml \
    --model runs/final_model.pth \
    --split test \
    --save-plots
```

评估结果包括：
- 总收益率、年化收益率
- 夏普比率、年化夏普比率
- 波动率、年化波动率
- 最大回撤
- 可视化图表

## 配置说明

### 默认配置 (`configs/default.yaml`)

```yaml
# 资产列表（16只美股）
assets: [AAPL, MSFT, NVDA, CVX, OXY, AAL, UAL, DAL, CCL, RCL, WYNN, LVS, AXP, BAC, JNJ, GOOGL]

# 数据分割（2014-2024）
split:
  train: ["2014-01-02", "2019-12-31"]  # 6年
  valid: ["2020-01-02", "2022-12-31"]  # 3年
  test:  ["2023-01-02", "2024-12-31"]  # 2年

# 交易参数
window: 20              # 状态窗口长度
fee_rate: 0.0005       # 手续费率 0.05%
buy_pct: 0.20          # 买入使用20%现金
sell_pct: 0.50         # 卖出50%持仓
initial_cash: 100000   # 初始资金 $100,000

# 模型参数
model:
  encoder: "TimeCNN"   # 编码器类型
  d_model: 128         # 模型维度
  n_layers: 3          # 层数

# 训练参数
train:
  total_env_steps: 1000000  # 总步数
  rollout_steps: 50         # Rollout步数
  gamma: 0.96               # 折扣因子
  entropy_coef: 0.05        # 熵系数
  lr_actor: 1.0e-5          # Actor学习率
  lr_critic: 1.0e-5         # Critic学习率
```

### 快速测试配置 (`configs/quick_test.yaml`)

用于快速验证代码的配置：
- 4只股票
- 2022-2024数据
- 50K训练步数
- 更小的模型

## 项目结构

```
QTMRL/
├── configs/                    # 配置文件
│   ├── default.yaml           # 默认配置
│   └── quick_test.yaml        # 快速测试配置
├── data/                       # 数据目录
│   ├── raw/                   # 原始数据
│   └── processed/             # 处理后的数据
├── qtmrl/                      # 核心代码
│   ├── indicators.py          # 技术指标计算
│   ├── dataset.py             # 数据集管理
│   ├── env.py                 # 交易环境
│   ├── models/                # 模型定义
│   │   ├── encoders.py        # 编码器
│   │   └── actor_critic.py    # Actor-Critic
│   ├── algo/                  # 算法实现
│   │   ├── rollout.py         # Rollout缓冲区
│   │   └── a2c.py             # A2C训练器
│   ├── eval/                  # 评估模块
│   │   ├── metrics.py         # 指标计算
│   │   ├── backtest.py        # 回测
│   │   └── plots.py           # 可视化
│   └── utils/                 # 工具函数
│       ├── seed.py            # 随机种子
│       ├── config.py          # 配置加载
│       ├── logging.py         # 日志记录
│       └── io.py              # 文件读写
├── scripts/                    # 运行脚本
│   ├── preprocess.py          # 数据预处理
│   ├── train.py               # 训练
│   └── evaluate.py            # 评估
├── tests/                      # 单元测试
├── README.md                   # 本文件
├── requirements.txt            # 依赖列表
└── setup.py                    # 安装脚本
```

## 在Google Colab上运行

### 方法1: 命令行脚本（推荐）

```python
# 1. 安装
!git clone https://github.com/yourusername/QTMRL.git
%cd QTMRL
!pip install -r requirements.txt

# 2. 数据预处理
!python scripts/preprocess.py --config configs/quick_test.yaml

# 3. 训练（使用GPU）
!python scripts/train.py --config configs/quick_test.yaml

# 4. 评估
!python scripts/evaluate.py \
    --config configs/quick_test.yaml \
    --model runs/final_model.pth \
    --split test \
    --save-plots

# 5. 查看结果
from IPython.display import Image, display
display(Image('results/test/portfolio_value.png'))
```

### 方法2: 挂载Google Drive保存结果

```python
from google.colab import drive
drive.mount('/content/drive')

# 将runs目录软链接到Drive
!ln -s /content/drive/MyDrive/QTMRL_runs runs
```

## 使用Wandb跟踪实验

1. 首次使用需要登录：

```python
import wandb
wandb.login()  # 会提示输入API key
```

2. 修改配置文件启用Wandb：

```yaml
logging:
  use_wandb: true
  wandb_project: "qtmrl"
  wandb_entity: "your-username"  # 可选
```

3. 运行训练，实验会自动上传到Wandb

## 技术细节

### 环境设计

- **状态空间**:
  - 特征窗口: `[W, N, F]` (W=窗口长度, N=资产数, F=特征数)
  - 持仓比例: `[N]`
  - 现金比例: `[1]`

- **动作空间**:
  - Factorized multi-head: 每个资产独立选择 `{SELL, HOLD, BUY}`
  - 非联合动作空间，避免组合爆炸

- **交易规则**:
  - BUY: 使用20%现金买入
  - SELL: 卖出50%持仓
  - 手续费: 0.05%（单边）
  - 禁止做空，禁止负现金

- **奖励函数**: 组合价值收益率 `r_t = (P_t / P_{t-1}) - 1`

### 模型架构

- **编码器**:
  - TimeCNN: 1D卷积 + 全局池化
  - Transformer: 多层自注意力机制

- **Actor**:
  - Multi-head架构，每个资产一个独立的head
  - 输出: `[N, 3]` 动作logits

- **Critic**:
  - 全局聚合（跨资产）
  - 输出: 标量状态价值

### A2C算法

- Rollout收集: 50步
- 优势函数: TD error
- 策略梯度 + 熵正则 + 价值函数
- 梯度裁剪: 1.0

## 数据说明

### 数据来源

使用 `yfinance` 从Yahoo Finance自动下载股票数据：
- 数据类型: 日频OHLCV（后复权）
- 时间范围: 2014-2024（默认配置）
- 股票数量: 16只美股（可配置）

### 技术指标

支持以下技术指标：
- **趋势指标**: SMA, EMA, Ichimoku
- **动量指标**: RSI, MACD
- **波动率指标**: ATR, Bollinger Bands, SuperTrend
- **形态指标**: Heikin-Ashi

### 数据预处理

1. 下载原始OHLCV数据
2. 计算技术指标
3. 按日期分割（train/valid/test）
4. Z-score标准化（仅在训练集上拟合）
5. 保存为numpy格式

## 常见问题

### Q1: 如何更换股票池？

修改配置文件中的 `assets` 列表：

```yaml
assets:
  - TSLA
  - AMZN
  - GOOG
  - META
```

### Q2: 如何调整训练时间？

修改 `total_env_steps`：

```yaml
train:
  total_env_steps: 500000  # 减少到50万步
```

### Q3: 训练时内存不足怎么办？

1. 减少资产数量
2. 减少模型维度 `d_model`
3. 使用更小的窗口 `window`
4. 缩短数据时间范围

### Q4: 如何添加新的技术指标？

在 `qtmrl/indicators.py` 中添加新函数，然后在配置文件中启用：

```yaml
features:
  indicators:
    your_indicator: [param1, param2]
```

### Q5: 如何实现消融实验？

1. 复制配置文件
2. 修改特定参数
3. 多次运行train.py
4. 比较results

## 性能基准

在默认配置下（16只股票，2014-2024数据）：

| 指标 | 训练集 | 验证集 | 测试集 |
|------|--------|--------|--------|
| 总收益率 | TBD | TBD | TBD |
| 夏普比率 | TBD | TBD | TBD |
| 最大回撤 | TBD | TBD | TBD |

> 注：实际性能取决于随机种子、市场环境等因素

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License

## 引用

如果使用本项目，请引用原始论文：

```bibtex
@article{qtmrl2024,
  title={QTMRL: Quantitative Trading with Multi-asset Reinforcement Learning},
  author={Your Name},
  year={2024}
}
```

## 联系方式

- 问题反馈: [GitHub Issues](https://github.com/yourusername/QTMRL/issues)
- 邮件: your.email@example.com

---

**免责声明**: 本项目仅供研究和学习使用，不构成任何投资建议。实际交易中使用本系统需自行承担风险。
