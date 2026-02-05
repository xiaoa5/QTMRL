# QTMRL 代码审查报告

**审查日期**: 2025-12-01
**审查范围**: 全部核心模块（环境、模型、算法、评估）
**总体评分**: 7/10

---

## 🔴 关键Bug（必须修复）

### 1. **代码重复问题** - CRITICAL
**位置**: `qtmrl/models/actor_critic.py`
**问题**: 整个文件内容重复了两次（第1-173行和第174-346行完全相同）

```python
# 文件结构：
Lines 1-173:   MultiHeadActor + Critic + create_models
Lines 174-346: MultiHeadActor + Critic + create_models (完全重复!)
```

**影响**:
- 代码维护困难
- 文件大小翻倍
- 可能导致混淆

**修复**: 删除第174-346行的重复内容

---

### 2. **Position Embedding未使用** - HIGH
**位置**: `qtmrl/models/encoders.py:199-206`

```python
# 当前代码 (BUG):
pos_emb = self.pos_embed(positions[:, i:i+1])  # 计算了
pos_emb = F.relu(pos_emb)
# ... 但下面只添加了pooled，pos_emb被丢弃了！
asset_encodings.append(pooled)  # ❌ pos_emb没有被使用

# 应该是:
asset_encodings.append(torch.cat([pooled, pos_emb], dim=-1))  # ✅
```

**影响**:
- 模型声称使用持仓信息，但实际上没有
- 决策时无法考虑当前持仓状态
- 这是一个**严重的功能缺失**

**修复**: 将position embedding连接到编码中

---

### 3. **Cash Embedding完全未使用** - MEDIUM
**位置**: `qtmrl/models/encoders.py:41-42`

```python
self.cash_embed = nn.Linear(1, d_model // 4)  # 定义了但从未调用
```

**影响**:
- 模型无法利用现金信息做决策
- 可能导致不合理的交易（现金不足时还想买入）

**修复**: 在forward中使用cash_embed，并与编码融合

---

### 4. **Entropy聚合不一致** - MEDIUM
**位置**: `qtmrl/models/actor_critic.py:116 vs 120`

```python
# Log概率是求和:
log_probs = log_probs_per_asset.sum(dim=-1)  # Sum across assets

# 但Entropy是平均:
entropy = entropy_per_asset.mean(dim=-1)     # Mean across assets ❌
```

**影响**:
- Entropy bonus的尺度与policy gradient不一致
- 可能影响训练稳定性

**修复**: 统一使用sum或统一使用mean

---

## 🟡 潜在问题（应该修复）

### 5. **Transformer位置编码大小限制**
**位置**: `qtmrl/models/encoders.py:256`

```python
self.pos_encoding = nn.Parameter(torch.randn(1, 100, d_model))
# ⚠️ 如果window > 100会崩溃
```

**修复**:
```python
max_window = 200  # 或从config读取
self.pos_encoding = nn.Parameter(torch.randn(1, max_window, d_model))
```

---

### 6. **组合价值计算时机设计选择**
**位置**: `qtmrl/env.py:138-181`

```python
# t时刻执行动作 (使用t时刻价格)
current_prices = self.Close[self.current_step]

# 但奖励基于t+1时刻的价格变化
next_prices = self.Close[next_step]
new_portfolio_value = self.cash + self.positions * next_prices
```

**分析**:
- ✅ **这个设计是正确的** - 模拟真实交易（今天交易，明天看收益）
- ⚠️ **但需要明确文档说明**

**建议**: 在代码注释中说明这是有意为之

---

### 7. **奖励函数过于简单**
**位置**: `qtmrl/env.py:185-189`

```python
reward = (new_portfolio_value / self.portfolio_value) - 1.0
# 只考虑收益率，不考虑风险
```

**建议**: 考虑风险调整的奖励
```python
# 可选方案:
# 1. Sharpe-like reward: return / volatility
# 2. Calmar ratio: return / max_drawdown
# 3. 惩罚波动性: return - λ * volatility
```

---

### 8. **Actor和Critic不共享编码器**
**位置**: `qtmrl/models/actor_critic.py:365-402`

```python
actor_encoder = TimeCNNEncoder(...)  # 独立编码器
critic_encoder = TimeCNNEncoder(...)  # 又创建一个
```

**影响**:
- 参数量翻倍
- 计算量翻倍
- 内存占用翻倍

**建议**: 共享编码器权重
```python
shared_encoder = TimeCNNEncoder(...)
actor = MultiHeadActor(encoder=shared_encoder, ...)
critic = Critic(encoder=shared_encoder, ...)
```

---

### 9. **Advantage标准化可能失败**
**位置**: `qtmrl/algo/a2c.py:155-156`

```python
if len(advantages) > 1:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # ⚠️ 小批量时std可能非常小，导致advantages爆炸
```

**建议**:
```python
if len(advantages) > 1:
    std = advantages.std()
    if std > 1e-3:  # 添加最小阈值
        advantages = (advantages - advantages.mean()) / std
```

---

## ✅ 做得好的地方

### 数据处理
- ✅ **无前瞻偏差** - 训练/验证/测试按时间顺序分割
- ✅ **正确的归一化** - 只在训练集上拟合，然后应用到验证/测试集
- ✅ **技术指标计算正确** - 全部使用历史数据

### 算法实现
- ✅ **Advantage计算正确** - 使用TD error
- ✅ **A2C实现合理** - 数学上正确
- ✅ **策略梯度计算正确**

### 评估指标
- ✅ **Sharpe ratio公式正确**
- ✅ **最大回撤计算正确**
- ✅ **年化因子正确** (252个交易日)

### 架构设计
- ✅ **良好的模块分离** - env / models / algo / eval 清晰分离
- ✅ **合理的状态表示** - 特征窗口 + 持仓 + 现金
- ✅ **因子化多头策略** - 每个资产独立动作头，合理

---

## 📋 修复优先级

### P0 - 立即修复（影响功能）
1. ⭐ **删除actor_critic.py中的重复代码**
2. ⭐ **修复position embedding未使用问题**
3. ⭐ **实现cash embedding功能**

### P1 - 尽快修复（影响训练）
4. 修复entropy聚合不一致
5. 添加Transformer位置编码大小检查
6. 改进advantage标准化

### P2 - 优化建议（提升性能）
7. 共享Actor-Critic编码器
8. 实现风险调整的奖励函数
9. 改进zero-padding策略
10. 添加数据质量检查

---

## 🔧 快速修复示例

### 修复 Position Embedding

**文件**: `qtmrl/models/encoders.py`

```python
# 当前 (line 206):
asset_encodings.append(pooled)

# 修复为:
asset_encodings.append(torch.cat([pooled, pos_emb], dim=-1))

# 同时修改输出维度 (line 51):
# 原来: self.fc_out = nn.Linear(d_model, d_model)
# 改为: self.fc_out = nn.Linear(d_model + d_model // 4, d_model)
```

### 修复 Cash Embedding

```python
# 在 forward 函数末尾 (line ~210):
# 1. 计算 cash embedding
cash_emb = self.cash_embed(cash)  # [B, d_model//4]
cash_emb = F.relu(cash_emb)

# 2. 与全局编码融合
global_enc = torch.cat([global_enc, cash_emb], dim=-1)  # [B, d_model + d_model//4]

# 3. 投影回 d_model
global_enc = self.fc_cash(global_enc)  # [B, d_model]

# 需要添加新层 (line ~44):
self.fc_cash = nn.Linear(d_model + d_model // 4, d_model)
```

---

## 📊 整体评估

| 维度 | 评分 | 说明 |
|------|------|------|
| **算法正确性** | 8/10 | 核心A2C算法实现正确 |
| **数据处理** | 9/10 | 无前瞻偏差，归一化正确 |
| **代码质量** | 5/10 | 有重复代码，部分功能未完成 |
| **模型设计** | 7/10 | 架构合理，但有优化空间 |
| **文档完整性** | 6/10 | 注释不够，设计选择未说明 |

**总评**: 7/10 - 核心逻辑正确，但有实现Bug需要修复

---

## 💡 建议

### 短期（本周）
1. 修复position/cash embedding bug
2. 删除重复代码
3. 修复entropy不一致

### 中期（两周内）
4. 共享编码器权重
5. 实现风险调整奖励
6. 完善单元测试

### 长期
7. 添加更多基线策略
8. 实现更复杂的编码器（GRU、Attention）
9. 支持更多资产类别

---

## ✅ 结论

**项目的核心逻辑是正确的**：
- ✅ 没有数据泄漏
- ✅ A2C算法实现正确
- ✅ 评估指标计算准确

**但有几个关键bug需要修复**：
- ❌ Position/cash embedding未使用
- ❌ 代码重复
- ⚠️ Entropy聚合不一致

**修复这些bug后，项目可以投入使用。**

建议优先修复P0级别的问题，然后再考虑性能优化。
