# QTMRL 论文复现与改进计划书

> 基于论文《QTMRL: An Agent for Quantitative Trading Decision-Making Based on Multi-Indicator Guided Reinforcement Learning》的复现与扩展

---

## 一、现状分析

### 1.1 当前代码库完成度评估

| 模块 | 完成状态 | 与论文一致性 | 备注 |
|------|----------|--------------|------|
| 数据下载与预处理 | ✅ 完成 | ⚠️ 部分一致 | 数据源和时间范围不同 |
| 技术指标计算 | ✅ 完成 | ✅ 一致 | 9种指标，覆盖趋势/波动/动量 |
| 交易环境 | ✅ 完成 | ⚠️ 部分一致 | 动作空间设计有差异 |
| A2C算法 | ✅ 完成 | ✅ 一致 | 核心算法实现正确 |
| 编码器 | ✅ 完成 | ➕ 扩展 | 论文未明确，你实现了CNN和Transformer |
| 评估指标 | ✅ 完成 | ⚠️ 需确认 | 计算方式可能有差异 |
| 基线策略 | ❌ 未完成 | - | 论文9个baseline均未实现 |
| 消融实验 | ❌ 未完成 | - | 论文5.3.2节为空 |
| 可视化分析 | ⚠️ 部分完成 | - | 基础图表有，高级分析缺失 |

### 1.2 论文与实现的关键差异

#### 差异1：动作空间设计

```
论文设计：
- 联合动作空间：2^N 个离散动作
- 每个动作编码为整数，表示N个资产的买/卖组合
- 16资产 = 65,536 个可能动作（存在组合爆炸）

当前实现：
- Factorized Multi-Head：每资产独立3动作（BUY/SELL/HOLD）
- 动作数量 = 3 × N = 48（线性增长）
- 更合理，但与论文不一致
```

**建议**：保留当前设计，在文档中说明这是对论文的改进。

#### 差异2：数据源与时间范围

```
论文：
- 数据源：Hugging Face jwigginton/timeseries-daily-sp500
- 时间范围：2000-01-03 至 2022-12-30（23年）
- 训练集：10年历史数据
- 测试集：2019-2021

当前实现：
- 数据源：Yahoo Finance (yfinance)
- 时间范围：2014-2024（默认配置）
- 分割：train 2014-2019, valid 2020-2022, test 2023-2024
```

#### 差异3：评估指标计算

```
论文：
- 对每只股票单独计算指标，然后取平均
- "metrics calculated based on the average price of the investment portfolio"

当前实现：
- 组合级别直接计算
- 基于总portfolio value
```

---

## 二、完整实施计划

### Phase 1: 数据与环境对齐（预计2-3天）

#### 任务1.1：统一数据源

**目标**：使用论文指定的数据源，确保可复现性

**具体步骤**：

1. 创建 `qtmrl/dataset_hf.py`，从Hugging Face下载数据：
   ```python
   # 使用 datasets 库
   from datasets import load_dataset
   dataset = load_dataset("jwigginton/timeseries-daily-sp500")
   ```

2. 修改 `configs/default.yaml`：
   ```yaml
   data:
     source: "huggingface"  # 或 "yfinance"
     hf_dataset: "jwigginton/timeseries-daily-sp500"
   split:
     train: ["2000-01-03", "2010-12-31"]  # 10年训练
     valid: ["2011-01-03", "2018-12-31"]  # 8年验证
     test:  ["2019-01-02", "2021-12-31"]  # 3年测试
   ```

3. 保留yfinance作为备选数据源

**验收标准**：
- [ ] 成功下载23年S&P 500数据
- [ ] 16只股票数据完整
- [ ] 数据预处理pipeline正常运行

#### 任务1.2：调整评估指标计算方式

**目标**：同时支持论文方式和组合方式

**具体步骤**：

1. 在 `qtmrl/eval/metrics.py` 中添加：
   ```python
   def calculate_per_asset_metrics(env, returns_per_asset):
       """论文方式：每资产计算后取平均"""
       pass
   
   def calculate_portfolio_metrics(portfolio_values):
       """组合方式：直接计算组合指标"""
       pass  # 当前已实现
   ```

2. 在配置中添加选项：
   ```yaml
   eval:
     metrics_mode: "per_asset"  # 或 "portfolio"
   ```

**验收标准**：
- [ ] 两种计算方式均可正常运行
- [ ] 结果可对比

---

### Phase 2: 基线策略实现（预计3-4天）

#### 任务2.1：规则策略实现

**目标**：实现4个规则基线

**文件结构**：
```
qtmrl/baselines/
├── __init__.py
├── base.py              # 基类定义
├── random_strategy.py   # 随机策略
├── ma_strategy.py       # 移动平均策略
├── dow_tracking.py      # 道琼斯跟踪策略
└── buy_and_hold.py      # 买入持有策略（额外添加）
```

**各策略规格**：

| 策略 | 动作规则 | 参数 |
|------|----------|------|
| Random | 20%买/20%卖/60%持有 | 随机种子42-46 |
| MA-10/20/30 | 价格上穿MA买入，下穿卖出 | 周期T=10/20/30 |
| Dow Tracking | 季度/年末再平衡 | 再平衡频率 |
| Buy & Hold | 初始等权买入，持有到底 | 无 |

**验收标准**：
- [ ] 每个策略可独立运行
- [ ] 输出格式与A2C一致（方便对比）
- [ ] Random策略5次运行结果稳定

#### 任务2.2：预测模型策略实现

**目标**：实现4个预测类基线（ARIMA, LSTM, CNN, ANN）

**设计思路**：
```
预测模型 → 预测下一日价格变化 → 转换为交易信号 → 执行交易

转换规则（论文ARIMA参数）：
- 预测涨幅 > 0.5% → BUY
- 预测跌幅 > 0.5% → SELL
- 其他 → HOLD
```

**文件结构**：
```
qtmrl/baselines/
├── predictive/
│   ├── __init__.py
│   ├── base_predictor.py    # 预测器基类
│   ├── arima_strategy.py    # ARIMA(5,1,0)
│   ├── lstm_strategy.py     # LSTM预测
│   ├── cnn_strategy.py      # CNN预测
│   └── ann_strategy.py      # ANN预测
```

**训练参数（与论文一致）**：
```yaml
predictive_baselines:
  common:
    optimizer: "adam"
    lr: 0.0001
    episodes: 100
    timesteps: 1000000
  arima:
    order: [5, 1, 0]
    threshold: 0.005
  lstm:
    hidden_size: 64
    num_layers: 2
  cnn:
    channels: [32, 64]
    kernel_size: 3
  ann:
    hidden_sizes: [128, 64]
```

**验收标准**：
- [ ] 每个模型可训练和推理
- [ ] 预测→交易信号转换正确
- [ ] 与A2C使用相同评估pipeline

#### 任务2.3：基线对比脚本

**目标**：一键运行所有基线对比

**创建** `scripts/compare_baselines.py`：
```python
# 功能：
# 1. 加载所有baseline
# 2. 在相同测试集上评估
# 3. 生成对比表格（类似论文Table 1/2）
# 4. 生成可视化对比图
```

**输出格式**：
```
results/baselines/
├── comparison_2019.csv
├── comparison_2020.csv
├── comparison_2021.csv
├── comparison_all.csv
├── figures/
│   ├── return_comparison.png
│   ├── sharpe_comparison.png
│   └── drawdown_comparison.png
```

**验收标准**：
- [ ] 生成与论文Table 1/2格式一致的表格
- [ ] 对比图清晰展示各策略差异

---

### Phase 3: 消融实验（预计2-3天）

#### 任务3.1：消融实验框架

**目标**：系统化测试各组件贡献

**创建** `configs/ablation.yaml`：
```yaml
ablation:
  # 实验维度
  dimensions:
    # 指标组合消融
    indicators:
      - name: "ohlcv_only"
        config:
          features:
            ohlcv: true
            indicators: {}
      - name: "ohlcv_trend"
        config:
          features:
            ohlcv: true
            indicators:
              sma: [10, 20, 50]
              ema: [12, 26]
      - name: "ohlcv_momentum"
        config:
          features:
            ohlcv: true
            indicators:
              rsi: [14]
              macd: [12, 26, 9]
      - name: "ohlcv_volatility"
        config:
          features:
            ohlcv: true
            indicators:
              atr: [14]
              bbands: [20, 2.0]
      - name: "all_indicators"
        config: null  # 使用默认完整配置
    
    # 窗口大小消融
    window_size:
      - 10
      - 20
      - 30
      - 60
    
    # 编码器消融
    encoder:
      - "TimeCNN"
      - "Transformer"
    
    # 熵系数消融
    entropy_coef:
      - 0.0
      - 0.01
      - 0.05
      - 0.1
  
  # 实验设置
  n_seeds: 3
  metrics: ["total_return", "sharpe", "volatility", "max_drawdown"]
```

**创建** `scripts/run_ablation.py`：
```python
# 功能：
# 1. 读取消融配置
# 2. 生成所有实验组合
# 3. 依次训练和评估
# 4. 收集结果到CSV
# 5. 生成消融分析图表
```

#### 任务3.2：消融结果分析

**输出**：
```
results/ablation/
├── indicator_ablation.csv
├── window_ablation.csv
├── encoder_ablation.csv
├── entropy_ablation.csv
├── figures/
│   ├── indicator_heatmap.png
│   ├── window_line_chart.png
│   └── ablation_summary.png
└── ablation_report.md
```

**验收标准**：
- [ ] 每个维度至少3个变体
- [ ] 每个变体运行3次取平均
- [ ] 生成清晰的对比图表

---

### Phase 4: 高级可视化与分析（预计2天）

#### 任务4.1：市场regime分析

**目标**：展示策略在不同市场状态下的表现

**关键时期**：
```python
MARKET_REGIMES = {
    "2008_crisis": ("2008-09-01", "2009-03-31"),      # 金融危机
    "2020_covid": ("2020-02-15", "2020-04-15"),       # COVID崩盘
    "2020_recovery": ("2020-04-15", "2020-12-31"),    # 疫情恢复
    "2021_bull": ("2021-01-01", "2021-12-31"),        # 牛市
}
```

**创建** `qtmrl/eval/regime_analysis.py`：
```python
def analyze_by_regime(portfolio_values, dates, regimes):
    """分regime计算指标"""
    pass

def plot_regime_comparison(results, save_path):
    """生成regime对比图"""
    pass
```

#### 任务4.2：持仓与交易分析

**新增可视化**：

1. **持仓热力图**：时间 × 资产的持仓比例矩阵
2. **交易频率统计**：每资产的买/卖/持有次数
3. **滚动指标曲线**：滚动Sharpe、滚动波动率
4. **资金流向图**：现金与各资产持仓的变化

**创建** `qtmrl/eval/advanced_plots.py`：
```python
def plot_position_heatmap(positions_history, asset_names, dates, save_path):
    """持仓热力图"""
    pass

def plot_rolling_metrics(portfolio_values, window=60, save_path=None):
    """滚动指标"""
    pass

def plot_capital_flow(cash_history, position_values, dates, save_path):
    """资金流向"""
    pass
```

**验收标准**：
- [ ] 生成至少4种高级可视化
- [ ] 图表清晰、专业、可用于论文

---

### Phase 5: 文档与报告（预计1-2天）

#### 任务5.1：自动化实验报告

**创建** `scripts/generate_report.py`：
```python
# 功能：
# 1. 收集所有实验结果
# 2. 生成Markdown报告
# 3. 包含所有图表和表格
# 4. 可选生成PDF
```

**报告模板**：
```markdown
# QTMRL 实验报告

## 1. 实验配置
- 数据范围
- 模型参数
- 训练设置

## 2. 基线对比
[Table 1: 2020年结果]
[Table 2: 2021年结果]
[对比图]

## 3. 消融实验
### 3.1 指标消融
### 3.2 窗口消融
### 3.3 编码器消融

## 4. 市场regime分析
### 4.1 金融危机时期
### 4.2 COVID时期

## 5. 结论
```

#### 任务5.2：更新项目文档

**更新文件**：
- `README.md`：添加复现说明和结果
- `CHANGELOG.md`：记录版本更新
- `ROADMAP.md`：更新完成状态

---

## 三、技术细节补充

### 3.1 需要新增的依赖

```txt
# requirements.txt 新增
datasets>=2.14.0        # Hugging Face数据集
statsmodels>=0.14.0     # ARIMA
optuna>=3.3.0           # 超参数优化（可选）
```

### 3.2 配置文件扩展

**创建** `configs/paper_reproduction.yaml`：
```yaml
# 严格按论文设置的配置
seed: 42

assets:
  - AAPL
  - MSFT
  - NVDA
  - CVX
  - OXY
  - AAL
  - UAL
  - DAL
  - CCL
  - RCL
  - WYNN
  - LVS
  - AXP
  - BAC
  - JNJ
  - GOOGL

data:
  source: "huggingface"
  hf_dataset: "jwigginton/timeseries-daily-sp500"

split:
  train: ["2000-01-03", "2010-12-31"]
  valid: ["2011-01-03", "2018-12-31"]
  test:  ["2019-01-02", "2021-12-31"]

window: 20
fee_rate: 0.0005
buy_pct: 0.20
sell_pct: 0.50
initial_cash: 10000  # 论文用$10,000

train:
  algo: "A2C"
  gamma: 0.96
  entropy_coef: 0.05
  value_coef: 1.0
  lr_actor: 1.0e-5
  lr_critic: 1.0e-5
  grad_clip: 1.0
  rollout_steps: 50
  total_env_steps: 1000000

eval:
  metrics_mode: "per_asset"  # 论文方式
```

### 3.3 目录结构最终形态

```
QTMRL/
├── configs/
│   ├── default.yaml
│   ├── quick_test.yaml
│   ├── paper_reproduction.yaml    # 新增
│   └── ablation.yaml              # 新增
├── data/
│   ├── raw/
│   └── processed/
├── qtmrl/
│   ├── __init__.py
│   ├── indicators.py
│   ├── dataset.py
│   ├── dataset_hf.py              # 新增
│   ├── env.py
│   ├── models/
│   │   ├── encoders.py
│   │   └── actor_critic.py
│   ├── algo/
│   │   ├── rollout.py
│   │   └── a2c.py
│   ├── baselines/                  # 新增
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── random_strategy.py
│   │   ├── ma_strategy.py
│   │   ├── dow_tracking.py
│   │   ├── buy_and_hold.py
│   │   └── predictive/
│   │       ├── __init__.py
│   │       ├── base_predictor.py
│   │       ├── arima_strategy.py
│   │       ├── lstm_strategy.py
│   │       ├── cnn_strategy.py
│   │       └── ann_strategy.py
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── backtest.py
│   │   ├── plots.py
│   │   ├── regime_analysis.py     # 新增
│   │   └── advanced_plots.py      # 新增
│   └── utils/
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── compare_baselines.py       # 新增
│   ├── run_ablation.py            # 新增
│   └── generate_report.py         # 新增
├── tests/
├── results/                        # 新增
│   ├── baselines/
│   ├── ablation/
│   └── reports/
└── docs/                           # 新增
    └── figures/
```

---

## 四、时间规划

### 总计预计：10-14天

```
Week 1:
├── Day 1-2: Phase 1 - 数据与环境对齐
├── Day 3-4: Phase 2.1 - 规则策略实现
└── Day 5-7: Phase 2.2 - 预测模型策略实现

Week 2:
├── Day 8-9: Phase 2.3 + Phase 3 - 基线对比与消融实验
├── Day 10-11: Phase 4 - 高级可视化
└── Day 12-14: Phase 5 - 文档与报告 + 调试
```

### 里程碑检查点

| 里程碑 | 日期 | 交付物 |
|--------|------|--------|
| M1 | Day 2 | 数据pipeline可运行 |
| M2 | Day 7 | 所有baseline可运行 |
| M3 | Day 9 | 完整对比表格生成 |
| M4 | Day 11 | 消融实验完成 |
| M5 | Day 14 | 完整报告生成 |

---

## 五、潜在风险与应对

### 风险1：Hugging Face数据集不可用

**应对**：保留yfinance作为备选，在README中说明两种数据源的使用方式。

### 风险2：预测模型训练时间过长

**应对**：
- 使用更小的模型配置进行快速验证
- 添加early stopping
- 考虑使用预训练好的模型直接评估

### 风险3：论文结果无法复现

**应对**：
- 论文Table 1/2本身是空的，无需严格对齐
- 重点是展示QTMRL相对于baseline的优势
- 记录所有实验参数，确保自己的结果可复现

### 风险4：动作空间差异导致性能差异

**应对**：
- 可以额外实现论文的联合动作空间版本作为对比
- 在文档中明确说明设计差异及理由

---

## 六、扩展方向（Phase 6+，可选）

完成论文复现后，可考虑以下扩展：

### 6.1 算法升级
- 实现PPO替代A2C
- 添加风险敏感奖励（Sharpe-aware, CVaR）
- 实现离线RL（IQL, CQL）

### 6.2 特征增强
- 整合你的FinTS-Representation表示学习
- 添加宏观特征（VIX等）
- 添加跨资产注意力机制

### 6.3 多智能体扩展
- 层级策略（配置+执行）
- 多agent协作

### 6.4 可解释性
- Attention可视化
- SHAP值分析
- 策略规则提取

---

## 七、验收清单

### 论文复现完成标准

- [ ] 数据：使用论文指定数据源和时间范围
- [ ] 基线：实现所有9个baseline
- [ ] 指标：支持论文的per-asset计算方式
- [ ] 表格：生成与论文Table 1/2格式一致的结果
- [ ] 消融：完成至少3个维度的消融实验
- [ ] 可视化：生成regime分析和持仓热力图
- [ ] 报告：自动生成完整实验报告

### 代码质量标准

- [ ] 所有新代码有docstring
- [ ] 关键函数有单元测试
- [ ] 配置化程度高，易于修改参数
- [ ] README更新，包含复现说明

---

## 附录A：论文核心参数速查

```yaml
# 论文Table - Training Details
initial_capital: 10000
fee_rate: 0.0005
window_size: 20
buy_pct: 0.20
sell_pct: 0.50
random_seed: 42

# A2C参数
a2c:
  optimizer: "adam"
  lr: 0.00001
  rollout_steps: 50
  gamma: 0.96
  entropy_coef: 0.05
  total_timesteps: 1000000

# Baseline参数
random:
  buy_prob: 0.20
  sell_prob: 0.20
  hold_prob: 0.60
  seeds: [42, 43, 44, 45, 46]

arima:
  order: [5, 1, 0]
  threshold: 0.005

lstm_cnn_ann:
  optimizer: "adam"
  lr: 0.0001
  loss: "cross_entropy"
  episodes: 100
  timesteps: 1000000
```

## 附录B：评估指标公式

```
Total Return Rate:
  Tr = (P_end - P_start) / P_start

Sharpe Ratio:
  Sr = E[r] / σ[r]
  
Volatility:
  Vol = σ[r] = std(returns)

Maximum Drawdown:
  Mdd = max((P_i - P_j) / P_i) for j > i
```

---

*文档版本：v1.0*
*最后更新：2024-12*
*作者：Yu*
