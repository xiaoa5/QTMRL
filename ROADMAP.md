# QTMRL 升级路线图

本文档规划了QTMRL项目的持续改进方向，分为短期、中期、长期三个阶段。

---

## 当前版本 v0.1.0 ✅

**已实现功能**：
- 基于A2C的多资产交易系统
- TimeCNN/Transformer编码器
- 完整的数据处理管道
- 基础评估指标和可视化
- Wandb集成

---

## 🎯 短期改进 (v0.2.0) - 1-2周

### 1. 基线对比 (Baseline Comparison)

**目标**：证明RL策略优于简单策略

**实现**：
- [ ] Buy & Hold 策略
- [ ] 等权重再平衡策略（每月/每季度）
- [ ] 动量策略（追涨杀跌）
- [ ] 均值回归策略

**文件**：`qtmrl/baselines/`
```python
# qtmrl/baselines/buy_and_hold.py
# qtmrl/baselines/rebalance.py
# qtmrl/baselines/momentum.py
# scripts/compare_baselines.py
```

**效果**：生成对比表格和图表
```
策略          总收益率  夏普比率  最大回撤
Buy & Hold    45.2%     0.82     -28.3%
等权重        38.7%     0.75     -25.1%
A2C (ours)    52.3%     1.15     -18.9%
```

### 2. 早停和模型选择 (Early Stopping & Model Selection)

**目标**：自动选择最优checkpoint，避免过拟合

**实现**：
- [ ] 基于验证集Sharpe ratio的早停
- [ ] 保存top-k模型
- [ ] 模型选择策略（最佳Sharpe vs 最小回撤）

**配置**：
```yaml
train:
  early_stopping:
    enabled: true
    patience: 5           # 验证集指标不提升的评估轮数
    metric: "sharpe"      # 监控指标
    mode: "max"           # 最大化
  save_top_k: 3           # 保存最好的3个模型
```

### 3. 风险敏感奖励 (Risk-Aware Rewards)

**目标**：平衡收益和风险

**实现多种奖励函数**：
```python
# 1. Sharpe-aware reward
r_t = (return_t - rf) / rolling_volatility

# 2. Drawdown penalty
r_t = return_t - lambda * drawdown_t

# 3. Sortino ratio (只惩罚下行波动)
r_t = return_t / downside_deviation

# 4. CVaR (条件风险价值)
r_t = return_t - alpha * CVaR_t
```

**配置**：
```yaml
reward:
  type: "sharpe_aware"    # return, sharpe_aware, drawdown_penalty, sortino, cvar
  params:
    lambda: 0.5           # 风险惩罚系数
    window: 20            # 滚动窗口
```

### 4. 消融实验自动化 (Automated Ablation)

**目标**：系统化地测试各组件的贡献

**实现**：
```bash
python scripts/ablation.py --config configs/ablation.yaml
```

**消融配置** (`configs/ablation.yaml`)：
```yaml
ablation:
  grid_search:
    window: [10, 20, 30, 60]
    encoder: ["TimeCNN", "Transformer"]
    indicators:
      - ["OHLCV"]                    # 只用OHLCV
      - ["OHLCV", "SMA", "EMA"]      # 添加趋势
      - ["OHLCV", "RSI", "MACD"]     # 添加动量
      - "all"                        # 全部指标
    entropy_coef: [0.0, 0.01, 0.05, 0.1]

  n_runs: 3                          # 每个配置运行3次
  output: "results/ablation.csv"
```

**输出**：生成热力图和表格

### 5. 增强可视化 (Enhanced Visualization)

**新增图表**：
- [ ] 持仓热力图（时间 x 资产）
- [ ] 月度/年度收益分布
- [ ] 滚动Sharpe比率曲线
- [ ] 风险归因分析
- [ ] 交易频率统计

```python
# qtmrl/eval/advanced_plots.py
plot_position_heatmap()
plot_rolling_sharpe()
plot_trade_frequency()
plot_risk_attribution()
```

### 6. 实验报告生成 (Experiment Report)

**目标**：一键生成完整的实验报告

```bash
python scripts/generate_report.py \
    --model runs/final_model.pth \
    --output reports/experiment_001.html
```

**报告内容**：
- 配置摘要
- 训练曲线
- 所有评估指标
- 可视化图表
- 基线对比
- 模型参数统计

---

## 🚀 中期改进 (v0.3.0) - 1-2个月

### 1. 更多RL算法 (Advanced RL Algorithms)

**目标**：对比不同算法的性能

**新增算法**：
- [ ] **PPO** (Proximal Policy Optimization) - 更稳定的策略梯度
- [ ] **SAC** (Soft Actor-Critic) - 连续动作空间 + 最大熵
- [ ] **TD3** (Twin Delayed DDPG) - 连续动作 + 低方差
- [ ] **Rainbow DQN** - 离散动作改进版

**结构**：
```
qtmrl/algo/
  ├── a2c.py           ✅
  ├── ppo.py           🆕
  ├── sac.py           🆕
  ├── td3.py           🆕
  └── rainbow.py       🆕
```

### 2. 连续动作空间 (Continuous Actions)

**目标**：直接输出资产权重分配

**动作定义**：
```python
# 离散动作 (当前)
action = [BUY, SELL, HOLD]  # 每个资产

# 连续动作 (新增)
action = [w1, w2, ..., wN]  # 权重 ∈ [0, 1], Σw_i ≤ 1
```

**优势**：
- 更精细的资金分配
- 避免频繁交易
- 更符合实际投资组合管理

**实现**：使用SAC或TD3算法

### 3. 注意力机制改进 (Advanced Attention)

**跨资产注意力**：
```python
class CrossAssetAttention(nn.Module):
    """资产间的相互影响建模"""
    def forward(self, asset_features):
        # asset_features: [B, N, D]
        # 计算资产间的attention权重
        attention_weights = self.attention(asset_features)
        # 聚合其他资产信息
        enhanced_features = attention_weights @ asset_features
        return enhanced_features
```

**时序注意力**：
```python
class TemporalAttention(nn.Module):
    """不同时间步的重要性"""
    def forward(self, temporal_features):
        # temporal_features: [B, W, D]
        # 学习不同历史时刻的重要性
        weights = self.attention(temporal_features)
        return weighted_sum(temporal_features, weights)
```

### 4. 特征工程增强 (Feature Engineering)

**新增特征类型**：

1. **宏观特征**：
   - VIX（波动率指数）
   - 利率数据
   - 汇率数据
   - 行业ETF

2. **市场微观结构**：
   - 日内高低价差
   - 成交量异常检测
   - 价格跳空

3. **情绪指标**：
   - 新闻情绪（需要API）
   - 社交媒体情绪
   - Put/Call比率

4. **横截面特征**：
   - 相对强弱（vs市场）
   - 行业内排名
   - 市值因子

```yaml
features:
  ohlcv: true
  technical_indicators: [...]
  macro:
    vix: true
    interest_rate: true
  sentiment:
    news: false        # 需要API
    social: false
  cross_sectional:
    market_relative: true
    sector_rank: true
```

### 5. 数据增强 (Data Augmentation)

**目标**：增加训练数据多样性，提高泛化能力

**方法**：
- [ ] 时间窗口随机裁剪
- [ ] 添加高斯噪声
- [ ] Bootstrap重采样
- [ ] 合成数据（GAN生成）

```python
# qtmrl/data/augmentation.py
class DataAugmentation:
    def random_crop(self, data, crop_ratio=0.9):
        """随机裁剪时间窗口"""
        pass

    def add_noise(self, data, noise_level=0.01):
        """添加高斯噪声"""
        pass

    def bootstrap_sample(self, data, n_samples=10):
        """Bootstrap重采样"""
        pass
```

### 6. 市场状态识别 (Market Regime Detection)

**目标**：识别不同市场状态，采用不同策略

**市场状态**：
- 牛市（Bull Market）
- 熊市（Bear Market）
- 震荡市（Sideways）
- 高波动（High Volatility）

**实现**：
```python
class MarketRegimeDetector:
    def detect_regime(self, market_data):
        """使用HMM或聚类识别市场状态"""
        pass

class RegimeAwarePolicy:
    """根据市场状态切换策略"""
    def __init__(self, policies):
        self.bull_policy = policies['bull']
        self.bear_policy = policies['bear']
        self.sideways_policy = policies['sideways']

    def select_action(self, state, regime):
        if regime == 'bull':
            return self.bull_policy(state)
        elif regime == 'bear':
            return self.bear_policy(state)
        else:
            return self.sideways_policy(state)
```

### 7. 超参数优化 (Hyperparameter Optimization)

**工具集成**：
- [ ] Optuna
- [ ] Ray Tune

```bash
python scripts/tune_hyperparams.py \
    --config configs/default.yaml \
    --n-trials 100 \
    --optimize sharpe
```

**搜索空间**：
```yaml
hyperparameters:
  lr_actor: [1e-6, 1e-4]         # log scale
  lr_critic: [1e-6, 1e-4]
  gamma: [0.90, 0.99]
  entropy_coef: [0.0, 0.1]
  d_model: [64, 128, 256]
  n_layers: [2, 3, 4, 5]
```

---

## 🔬 长期研究方向 (v0.4.0+) - 3-6个月

### 1. 离线强化学习 (Offline RL)

**动机**：利用历史数据，避免在线探索风险

**算法**：
- [ ] Conservative Q-Learning (CQL)
- [ ] Batch Constrained Q-learning (BCQ)
- [ ] Implicit Q-Learning (IQL)

**优势**：
- 无需实时交互
- 可利用大规模历史数据
- 适合真实交易场景

### 2. 多智能体强化学习 (Multi-Agent RL)

**场景**：多个agent管理不同资产组合

**方法**：
- [ ] Independent Q-Learning
- [ ] QMIX
- [ ] MADDPG

**应用**：
- 协作：多个策略投票
- 竞争：模拟多方博弈

### 3. 模型可解释性 (Interpretability)

**目标**：理解模型决策依据

**方法**：
- [ ] Attention可视化
- [ ] SHAP值分析
- [ ] 特征重要性
- [ ] 反事实解释

```python
# qtmrl/explainability/
explain_action(model, state)  # 为什么选择BUY?
visualize_attention()         # 关注哪些时间步？
feature_importance()          # 哪些指标最重要？
```

### 4. 实时交易接口 (Live Trading Interface)

**警告**：实盘交易风险极高，需谨慎！

**架构**：
```
qtmrl/live/
  ├── broker.py           # 券商接口抽象
  ├── alpaca.py           # Alpaca接口
  ├── interactive_brokers.py
  ├── paper_trading.py    # 模拟盘
  └── live_agent.py       # 实时agent
```

**功能**：
- [ ] 实时数据流
- [ ] 订单管理
- [ ] 风险控制（止损、仓位限制）
- [ ] 监控和报警

### 5. 分布式训练 (Distributed Training)

**目标**：加速训练，支持大规模实验

**框架**：
- [ ] Ray RLlib
- [ ] PyTorch Distributed

**功能**：
- 多GPU训练
- 分布式rollout收集
- 异步参数更新

### 6. 元学习 (Meta-Learning)

**目标**：快速适应新市场环境

**方法**：
- [ ] MAML (Model-Agnostic Meta-Learning)
- [ ] Reptile

**应用**：
- 快速适应新股票
- 跨市场迁移（美股 → 港股）
- 少样本学习

### 7. 生成式AI增强 (Generative AI)

**应用场景**：

1. **新闻情绪分析**：
   - 使用LLM分析财经新闻
   - 提取关键信息作为特征

2. **市场解说**：
   - 生成交易决策解释
   - 自动撰写投资报告

3. **策略生成**：
   - 用LLM生成交易策略代码
   - 自动化策略backtesting

```python
# 示例：LLM辅助决策
sentiment = llm.analyze_news(news_text)
explanation = llm.explain_action(state, action)
```

### 8. 集成学习 (Ensemble Methods)

**方法**：
- [ ] 多个独立模型投票
- [ ] Bagging (Bootstrap Aggregating)
- [ ] Boosting
- [ ] Stacking

**实现**：
```python
class EnsembleAgent:
    def __init__(self, agents):
        self.agents = agents

    def act(self, state):
        actions = [agent.act(state) for agent in self.agents]
        return majority_vote(actions)  # 或加权平均
```

### 9. 对抗鲁棒性 (Adversarial Robustness)

**目标**：提高模型在异常市场的鲁棒性

**方法**：
- [ ] 对抗训练
- [ ] Domain Randomization
- [ ] 鲁棒优化

```python
# 添加对抗样本训练
def adversarial_training(model, state):
    # 生成对抗样本
    perturbed_state = add_adversarial_noise(state)
    # 在对抗样本上训练
    loss = compute_loss(model, perturbed_state)
    return loss
```

---

## 📊 实验管理和工程优化

### 1. 完整的MLOps流程

**工具链**：
- [ ] **实验跟踪**: Wandb / MLflow
- [ ] **模型注册**: Model Registry
- [ ] **版本控制**: DVC (Data Version Control)
- [ ] **CI/CD**: GitHub Actions

**流程**：
```
代码修改 → 自动测试 → 训练 → 评估 → 模型注册 → 部署
```

### 2. Web可视化界面

**功能**：
- [ ] 实时监控训练
- [ ] 交互式回测
- [ ] 参数调整和重新训练
- [ ] 策略对比

**技术栈**：
- Streamlit / Gradio
- Plotly交互图表

### 3. Docker容器化

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "scripts/train.py"]
```

```bash
# 一键运行
docker-compose up
```

---

## 🎓 研究论文方向

如果要发表论文，可以探索：

1. **新奖励函数设计**：如何平衡收益和风险？
2. **市场状态自适应**：在不同市场条件下的策略切换
3. **跨市场迁移学习**：美股经验迁移到A股
4. **可解释性研究**：为什么RL策略有效？
5. **离线RL在金融中的应用**：如何利用历史数据？
6. **对抗鲁棒性**：如何应对黑天鹅事件？

---

## 📅 实施建议

### 优先级排序

**高优先级**（立即实施）：
1. ✅ 基线对比 - 证明RL有效性
2. ✅ 早停和模型选择 - 避免过拟合
3. ✅ 风险敏感奖励 - 实用性改进
4. ✅ 消融实验自动化 - 系统化研究

**中优先级**（1-2个月）：
5. PPO算法 - 更稳定训练
6. 连续动作空间 - 更灵活策略
7. 注意力机制改进 - 性能提升
8. 特征工程增强 - 信息增益

**低优先级**（长期研究）：
9. 离线RL - 研究前沿
10. 实时交易 - 实盘应用
11. 元学习 - 高级主题

### 迭代开发流程

```
1. 选择一个改进方向
   ↓
2. 在quick_test配置上验证
   ↓
3. 在完整配置上实验
   ↓
4. 记录结果，更新文档
   ↓
5. 提交代码，发布新版本
   ↓
6. 选择下一个方向
```

---

## 🤝 贡献指南

欢迎贡献！可以从以下方面入手：

1. **实现一个新算法** (PPO, SAC等)
2. **添加新的基线策略**
3. **改进可视化**
4. **优化性能**（速度、内存）
5. **编写教程和文档**

---

## 📮 反馈和建议

如果你有新的想法或建议，欢迎：
- 提交GitHub Issue
- 发起Discussion
- 提交Pull Request

让我们一起把QTMRL打造成最好的量化交易RL框架！🚀
