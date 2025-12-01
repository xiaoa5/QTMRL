# QTMRL 实施计划 - 完善版

> 基于 Yu 的原始计划，增加技术细节和优先级建议

---

## 📌 关键修正

### 修正1：Phase 1 时间调整

**原计划**：2-3天
**建议调整**：3-5天

**理由**：
1. Hugging Face数据集可能需要数据清洗和格式转换
2. 需要验证16只股票的数据完整性
3. 需要与现有yfinance pipeline兼容

**建议增加检查点**：
```bash
# Day 1: 数据探索
python scripts/explore_hf_data.py --dataset jwigginton/timeseries-daily-sp500

# Day 2: 数据下载和清洗
python scripts/preprocess.py --config configs/paper_reproduction.yaml --source huggingface

# Day 3: 数据验证
python scripts/validate_data.py --compare-yfinance
```

---

### 修正2：Phase 2 优先级调整

**建议顺序**：

1. **先实现简单基线**（1-2天）
   - Buy & Hold
   - Random
   - MA策略

   ✅ **立即可用，快速验证系统**

2. **再实现预测模型**（3-5天）
   - ARIMA（最简单）
   - LSTM
   - CNN/ANN（可选）

   ⚠️ **这部分可能比预期复杂**

**降低复杂度建议**：
```python
# 简化版预测模型策略
class SimplePredictiveStrategy:
    """使用预训练模型或简单逻辑"""

    def __init__(self, model_type='arima'):
        if model_type == 'arima':
            # 使用statsmodels的自动ARIMA
            self.model = auto_arima(...)
        elif model_type == 'lstm':
            # 使用简单的单层LSTM
            self.model = SimpleLSTM(input_size=5, hidden_size=32)
```

---

### 修正3：添加快速验证路径

**建议增加 Phase 0: 快速验证（1天）**

```python
# 目标：验证核心功能可用
# scripts/quick_validation.py

def validate_current_system():
    """验证当前系统基础功能"""

    # 1. 数据pipeline测试
    assert preprocess_runs_successfully()

    # 2. 训练基础测试（100步）
    assert train_runs_successfully(steps=100)

    # 3. 评估测试
    assert evaluate_runs_successfully()

    # 4. 输出测试报告
    generate_validation_report()
```

**运行**：
```bash
python scripts/quick_validation.py
# 预期输出：
# ✓ 数据预处理正常
# ✓ 模型训练正常
# ✓ 评估流程正常
# → 系统基础功能验证通过，可以开始复现
```

---

## 🎯 优先级建议

### Tier 1：核心功能（必须完成）

| 任务 | 预估时间 | 优先级 | 理由 |
|------|----------|--------|------|
| Phase 0: 快速验证 | 1天 | 🔴 P0 | 确保基础可用 |
| Phase 1: 数据对齐 | 3-4天 | 🔴 P0 | 复现基础 |
| Phase 2.1: 简单基线 | 2天 | 🔴 P0 | Buy&Hold, Random |
| Phase 3: 基础消融 | 2天 | 🟠 P1 | 指标和窗口消融 |

**总计：8-9天**

### Tier 2：增强功能（强烈建议）

| 任务 | 预估时间 | 优先级 |
|------|----------|--------|
| Phase 2.2: ARIMA基线 | 2天 | 🟠 P1 |
| Phase 4: 基础可视化 | 1天 | 🟠 P1 |
| Phase 5: 基础报告 | 1天 | 🟠 P1 |

**总计：4天**

### Tier 3：完善功能（时间允许）

| 任务 | 预估时间 | 优先级 |
|------|----------|--------|
| Phase 2.2: LSTM/CNN/ANN | 3-4天 | 🟡 P2 |
| Phase 4: 高级可视化 | 1-2天 | 🟡 P2 |
| Phase 3: 完整消融 | 2天 | 🟡 P2 |

---

## 🛠 技术细节补充

### 补充1：Hugging Face数据集处理

**潜在问题**：
```python
# 论文数据集可能的格式
{
    'date': '2020-01-01',
    'symbol': 'AAPL',
    'open': 100.0,
    'high': 105.0,
    'low': 98.0,
    'close': 103.0,
    'volume': 1000000,
    'adjusted_close': 103.0  # 可能有或没有
}
```

**建议实现**：
```python
# qtmrl/dataset_hf.py

from datasets import load_dataset
import pandas as pd

class HuggingFaceDataset:
    """Hugging Face数据集加载器"""

    def __init__(self, dataset_name, assets):
        self.dataset_name = dataset_name
        self.assets = assets

    def load(self):
        """加载数据并转换为标准格式"""
        try:
            # 尝试加载数据集
            dataset = load_dataset(self.dataset_name)

            # 检查数据格式
            print(f"Dataset keys: {dataset.keys()}")
            print(f"Features: {dataset['train'].features}")

            # 转换为pandas DataFrame
            df = dataset['train'].to_pandas()

            # 标准化列名
            df = self._standardize_columns(df)

            # 筛选指定资产
            df = df[df['symbol'].isin(self.assets)]

            return df

        except Exception as e:
            print(f"⚠️ Hugging Face数据加载失败: {e}")
            print("→ 回退到yfinance")
            return None

    def _standardize_columns(self, df):
        """标准化列名为 OHLCV 格式"""
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'adjusted_close': 'Adj Close',
        }
        return df.rename(columns=column_mapping)
```

### 补充2：简化的ARIMA策略

**完整实现参考**：
```python
# qtmrl/baselines/arima_strategy.py

from statsmodels.tsa.arima.model import ARIMA
import numpy as np

class ARIMAStrategy:
    """ARIMA预测策略"""

    def __init__(self, order=(5,1,0), threshold=0.005):
        self.order = order
        self.threshold = threshold
        self.models = {}  # 每个资产一个模型

    def train(self, historical_prices, asset_name):
        """训练ARIMA模型"""
        try:
            model = ARIMA(historical_prices, order=self.order)
            fitted = model.fit()
            self.models[asset_name] = fitted
            return True
        except:
            print(f"⚠️ ARIMA训练失败: {asset_name}")
            return False

    def predict(self, asset_name, steps=1):
        """预测未来价格变化"""
        if asset_name not in self.models:
            return 0.0

        forecast = self.models[asset_name].forecast(steps=steps)
        return forecast[0]

    def get_action(self, current_price, predicted_price):
        """根据预测转换为交易信号"""
        change = (predicted_price - current_price) / current_price

        if change > self.threshold:
            return Action.BUY
        elif change < -self.threshold:
            return Action.SELL
        else:
            return Action.HOLD

    def run_backtest(self, env, window=60):
        """运行回测"""
        state = env.reset()
        done = False

        while not done:
            actions = []
            for i, asset in enumerate(env.assets):
                # 获取历史价格
                hist_prices = env.get_price_history(asset, window)

                # 训练/更新模型
                self.train(hist_prices, asset)

                # 预测并决策
                current_price = hist_prices[-1]
                predicted = self.predict(asset)
                action = self.get_action(current_price, predicted)

                actions.append(action)

            state, reward, done, info = env.step(np.array(actions))

        return env.get_portfolio_values()
```

### 补充3：评估指标 - 两种计算方式

```python
# qtmrl/eval/metrics.py

def calculate_metrics(env, mode='portfolio'):
    """
    计算评估指标

    Args:
        env: 交易环境
        mode: 'portfolio' 或 'per_asset'
    """
    if mode == 'portfolio':
        return calculate_portfolio_metrics(env)
    elif mode == 'per_asset':
        return calculate_per_asset_metrics(env)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def calculate_portfolio_metrics(env):
    """当前实现：组合级别"""
    pv = env.get_portfolio_values()
    returns = np.diff(pv) / pv[:-1]

    return {
        'total_return': (pv[-1] / pv[0]) - 1,
        'sharpe': np.mean(returns) / (np.std(returns) + 1e-8),
        'volatility': np.std(returns),
        'max_drawdown': calculate_max_drawdown(pv)
    }

def calculate_per_asset_metrics(env):
    """论文方式：每资产计算后平均"""
    assets_metrics = []

    for asset_idx in range(env.n_assets):
        # 获取该资产的价格历史
        asset_prices = env.get_asset_price_history(asset_idx)

        # 获取该资产的持仓历史
        asset_positions = env.get_asset_position_history(asset_idx)

        # 计算该资产的returns
        asset_values = asset_prices * asset_positions
        asset_returns = np.diff(asset_values) / (asset_values[:-1] + 1e-8)

        # 计算指标
        metrics = {
            'total_return': (asset_values[-1] / asset_values[0]) - 1,
            'sharpe': np.mean(asset_returns) / (np.std(asset_returns) + 1e-8),
            'volatility': np.std(asset_returns),
            'max_drawdown': calculate_max_drawdown(asset_values)
        }
        assets_metrics.append(metrics)

    # 平均所有资产的指标
    return {
        key: np.mean([m[key] for m in assets_metrics])
        for key in assets_metrics[0].keys()
    }
```

---

## 📋 实施建议

### Week 1: 核心功能（Tier 1）

```
Day 1:  Phase 0 - 快速验证
Day 2-3: Phase 1.1 - HF数据集 (优先yfinance后备)
Day 4:  Phase 1.2 - 评估指标两种模式
Day 5-6: Phase 2.1 - Buy&Hold, Random, MA策略
Day 7:  Phase 3 - 基础消融（指标、窗口）

✅ Checkpoint: 能生成基本的对比表格
```

### Week 2: 增强功能（Tier 2 + 部分Tier 3）

```
Day 8-9:  Phase 2.2 - ARIMA策略
Day 10:   Phase 4 - 基础可视化
Day 11:   Phase 5 - 报告生成
Day 12-14: Buffer - 调试、完善、文档

✅ Checkpoint: 完整的复现报告
```

### 可选Week 3: 完善功能（Tier 3）

```
根据时间和精力决定：
- LSTM/CNN/ANN策略
- 高级可视化
- 更多消融维度
```

---

## ⚡ 快速启动路径

如果想**快速看到结果**，建议按这个顺序：

1. **Day 1**: 运行现有系统，确保能跑通
   ```bash
   python tests/test_imports.py
   python scripts/preprocess.py --config configs/quick_test.yaml
   python scripts/train.py --config configs/quick_test.yaml
   ```

2. **Day 2-3**: 实现Buy & Hold基线
   ```python
   # 最简单的基线，30分钟就能写完
   class BuyAndHoldStrategy:
       def run_backtest(self, env):
           # 第一天买入
           initial_actions = [Action.BUY] * env.n_assets
           env.step(initial_actions)

           # 之后全部HOLD
           while not done:
               actions = [Action.HOLD] * env.n_assets
               state, reward, done, info = env.step(actions)

           return env.get_portfolio_values()
   ```

3. **Day 3-4**: 生成第一张对比表
   ```bash
   python scripts/compare_baselines.py \
       --strategies buy_and_hold random ma_10 qtmrl \
       --output results/first_comparison.csv
   ```

   ✅ **成就感爆棚！**

---

## 🎓 学习建议

### 如果是为了论文发表

**优先级**：
1. ✅ 基线对比（核心贡献点）
2. ✅ 消融实验（证明设计有效）
3. ✅ 可视化分析（图表专业）
4. ⚠️ 预测模型（时间允许再做）

### 如果是为了学习RL

**优先级**：
1. ✅ 理解A2C算法
2. ✅ 实验不同超参数
3. ✅ 可视化训练过程
4. ✅ 尝试改进奖励函数

---

## 🔍 额外建议

### 1. 添加渐进式验证

在每个Phase完成后，添加验证脚本：

```python
# scripts/validate_phase1.py
def validate_phase1():
    # 检查数据加载
    assert data_can_be_loaded()
    # 检查16只股票完整
    assert all_16_stocks_present()
    # 检查时间范围正确
    assert date_range_matches_paper()

    print("✅ Phase 1 验证通过")

# scripts/validate_phase2.py
def validate_phase2():
    # 检查所有基线可运行
    for strategy in ['buy_hold', 'random', 'ma']:
        assert strategy_runs(strategy)

    print("✅ Phase 2 验证通过")
```

### 2. 添加性能基准

记录每个阶段的运行时间：

```python
BENCHMARKS = {
    'preprocess': '< 5分钟',
    'train_100k_steps': '< 30分钟 (GPU)',
    'evaluate': '< 2分钟',
    'generate_report': '< 1分钟'
}
```

### 3. 创建问题追踪

使用GitHub Issues或简单的TODO文件：

```markdown
# issues.md

## 已知问题
- [ ] HF数据集列名可能不一致
- [ ] ARIMA在某些股票上收敛慢
- [ ] 长时间训练内存占用大

## 待确认
- [ ] 论文的per-asset指标具体计算方式
- [ ] 动作空间的确切定义
```

---

## 总结

你的计划非常扎实！主要建议：

1. **增加Phase 0**：快速验证现有系统
2. **调整Phase 1时间**：3-5天更合理
3. **Phase 2分优先级**：先简单后复杂
4. **添加检查点**：每个阶段可验证
5. **提供快速路径**：3-4天就能看到初步结果

**最重要的是**：不要一开始就追求完美，先跑通核心流程，再逐步完善！🚀

需要我帮你开始实现哪个部分吗？
