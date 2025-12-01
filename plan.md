
# 0) 总体假设与目标

**目标**：在日K数据上复现一个**多资产 A2C 交易 agent**，输入为**多指标的时间窗**，输出为**每只股票的离散动作（买/卖/停）**，**共享资金池**，组合级奖励，评估涵盖 **收益/Sharpe/波动/MDD**，并提供清晰、可扩展的代码结构与配置。

**关键一致性**（与论文）

* 状态：(s_t \in \mathbb{R}^{W\times N\times F})，含持仓/现金（portfolio status）
* 动作：**factorized multi-head**（每股一头，非 (3^N) 联合软最大）
* 交易规则：买=20%现金，卖=50%持仓，手续费=0.05%（单边），资金不足则跳过
* 算法：A2C（policy/value/entropy），(\gamma=0.96)，entropy coef=0.05，rollout=50，训练步数≈1e6
* 评估：组合级指标（对 16 只股票一起回测）

---

# 1) 仓库结构与配置

**产出物**：标准化代码结构 + 单一 YAML 配置，便于替换组件与做消融。

```
qtmrl/
  configs/
    default.yaml
  data/
    raw/            # 原始日K与元数据
    processed/      # 计算好指标后的 parquet/hdf5
    splits.json     # 训练/验证/测试日期范围
  qtmrl/
    indicators.py
    dataset.py
    env.py
    models/
      encoders.py
      actor_critic.py
    algo/
      a2c.py
      rollout.py
    eval/
      metrics.py
      backtest.py
      plots.py
    utils/
      seed.py
      io.py
      logging.py
      config.py
  scripts/
    preprocess.py
    train.py
    evaluate.py
    ablation.py
  README.md
```

**默认配置（configs/default.yaml）**（Claude 直接写）：

```yaml
seed: 42
assets: [AAPL, MSFT, NVDA, CVX, OXY, AAL, UAL, DAL, CCL, RCL, WYNN, LVS, AXP, BAC, JNJ, GOOGL]
window: 20
fee_rate: 0.0005
buy_pct: 0.20
sell_pct: 0.50
split:
  train: ["2000-01-03", "2010-12-31"]
  valid: ["2011-01-03", "2018-12-31"]
  test:  ["2019-01-02", "2021-12-31"]
features:
  ohlcv: true
  indicators:
    sma: [10, 20, 50]
    ema: [12, 26]
    rsi: [14]
    macd: [12, 26, 9]     # (fast, slow, signal)
    atr: [14]
    bbands: [20, 2.0]     # (period, std)
    ichimoku: [9, 26, 52]
    heikin_ashi: true
    supertrend: [10, 3.0] # (period, multiplier)
normalization:
  method: zscore
  fit_on: "train"
  include: ["OHLCV", "indicators"]
state:
  include_pos: true
  include_cash: true
model:
  encoder: "TimeCNN"     # or "Transformer"
  d_model: 128
  n_layers: 3
  dropout: 0.0
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
  log_interval_steps: 1000
eval:
  deterministic: true
  use_fee: true
  metrics: ["total_return","sharpe","volatility","max_drawdown"]
logging:
  use_wandb: false
  output_dir: "./runs"
```

**Claude Prompt（创建骨架与配置）**

> 请创建上述目录与文件骨架，并填入 `configs/default.yaml` 内容。为 Python 包写好 `__init__.py`。

---

# 2) 数据与特征工程

**目标**：下载/加载日K（OHLCV），按资产对齐，计算指标，做 z-score 标准化（**仅用训练集拟合 (\mu,\sigma)**），保存为列式文件（parquet/hdf5）。

**关键决策与默认值**

* 价格：使用**后复权 Close**（若不可得，则用原始 Close 并接受跳空）
* 缺失：价格前向填充；volume 缺失→0；指标不可计算的前 W 天**丢弃**
* 标准化：对每个特征维按（训练集的）全时间×全资产联合统计做 z-score

**输出张量**

* `X[t, n, f]`：标准化后特征
* `Close[t, n]`：未标准化 close（成交与估值）
* `dates[t]`

**Claude Prompt（数据处理脚本）**

> 在 `scripts/preprocess.py` 中实现：
>
> 1. 读取原始日K（可先用 CSV 占位，后续替换为 HF 数据），按 `assets` 子集对齐；
> 2. 计算配置里列出的指标（在 `qtmrl/indicators.py`）；
> 3. 合并成特征张量并做 z-score（使用训练集范围拟合）；
> 4. 丢弃指标无效的起始段；
> 5. 保存到 `data/processed/`（parquet 或 hdf5）；
> 6. 在 `data/splits.json` 写入日期分割。
>    请写完整、可运行代码，并有日志输出维度/资产数/特征数。

---

# 3) 环境（Env）设计

**目标**：实现**共享资金池**的多资产日度交易环境，**factorized 动作**、**组合级奖励**、**费用**、**仓位与现金**随时间演化。

**状态**
[
s_t = \big[X_{t-W+1:t}\in \mathbb{R}^{W\times N\times F},\ \text{pos}_t\in \mathbb{R}^N,\ \text{cash}_t\in \mathbb{R}\big]
]

**动作（每股一头）**
(a_{t,i} \in {\text{BUY},\ \text{SELL},\ \text{HOLD}})

**执行规则**

* BUY：用 `buy_pct * cash_t` 下单，若资金不足则跳过
* SELL：卖出 `sell_pct * pos_{t,i}` 的股数
* 费用：`fee = amount * fee_rate`（单边）
* 禁止做空，禁止负现金

**奖励**
[
P_t = \text{cash}*t + \sum*{i=1}^{N}\text{pos}*{t,i}\cdot \text{price}*{t,i}
]
[
r_t = \frac{P_t}{P_{t-1}} - 1
]

**Claude Prompt（环境实现）**

> 在 `qtmrl/env.py` 实现 `TradingEnv`：
>
> * `reset()` 返回初始 state（从 `window` 开始）
> * `step(action_vec)` 接受长度为 N 的整数数组，执行买卖与费用结算，返回 `(state', reward, done, info)`，其中 `info['pv'] = P_t`；
> * 内部维护 `cash`, `pos[N]`, `portfolio_value`；
> * 提供 `get_state_tensor()`，把 `X_window`, `pos`, `cash` 组合为可馈入模型的张量。
>   加上单元测试：构造一个两资产小样，验证买卖逻辑、费用与 PV 计算。

---

# 4) 模型（Encoder + Multi-Head Actor + Critic）

**Baseline Encoder：TimeCNN（简单稳健）**

* 对每个资产的 (W\times F) 用 1D Conv（沿时间维），池化到向量 (z_i\in\mathbb{R}^{D})
* 拼接 `pos[i]` 与 `cash_norm` 到每个 (z_i)（或 cash 走全局通道）
* **Cross-Asset 层**（简单版）：把 (z_i) 堆叠后做一层 self-attention 或者 MLP

**Actor（multi-head）**

* 对每个 (z_i) 经过 head MLP 输出 `logits[i, 3]`（buy/sell/hold）
* 输出 `pi[i] = softmax(logits[i])`

**Critic**

* 用**全局聚合**（例如 mean/attention pooling）得到 (z_{global})，MLP 输出标量 (V(s))

**Claude Prompt（模型实现）**

> 在 `qtmrl/models/encoders.py` 实现 `TimeCNNEncoder`（Conv1d/GLU/池化）；
> 在 `qtmrl/models/actor_critic.py` 实现：
>
> * `MultiHeadActor(encoder, d_model, n_assets)`：前向返回 `pi`（N×3）、`logits`；
> * `Critic(encoder)`：前向返回 `V(s)`；
> * 注意 `encoder` 可共享，也可各自实例化（从配置读取）。
>   写 `forward()` 明确输入/输出形状，附上 shape assert。

---

# 5) 算法（A2C）

**损失**
[
A_t = G_t - V(s_t),\quad G_t = r_t + \gamma G_{t+1}
]
[
\log\pi(a_t|s_t) = \sum_{i=1}^{N} \log\pi_i(a_{t,i}|s_t)
]
[
L = -\sum_t \log\pi(a_t|s_t)A_t ;+; c_v\sum_t\big(V(s_t)-G_t\big)^2 ;-; c_e\sum_t H(\pi(\cdot|s_t))
]

**实现要点**

* rollout 缓冲区：保存 `state, logits, action_vec, reward, value, last_state`
* bootstrap：episode 末尾 `last_value = 0`，否则 `V(s_{T})`
* 优势、回报向后递推
* 梯度裁剪（1.0）
* entropy 系数可常量（默认 0.05）

**Claude Prompt（A2C 实现）**

> 在 `qtmrl/algo/rollout.py` 写 `RolloutBuffer`；
> 在 `qtmrl/algo/a2c.py` 写 `A2CTrainer`：
>
> * `collect_rollout(env, actor, critic, rollout_steps)`
> * `compute_returns_advantages(rewards, values, gamma)`
> * `update_actor_critic(buffer, entropy_coef, value_coef, grad_clip)`
> * 记录标量到 logger（loss, entropy, value_loss, avg_reward）。
>   在 `scripts/train.py` 编写训练循环：加载配置/数据→构建 env/model/trainer→迭代至 total_env_steps。

---

# 6) 评估与回测

**评估设置**

* **deterministic**：对每头取 `argmax(pi[i])`
* 保存：组合净值曲线、每日收益、4 指标
* 指标：

  * 总收益率：(\frac{P_{end}}{P_{start}}-1)
  * 波动率：(\sigma[r])（日频），年化可乘 (\sqrt{252})（可选）
  * Sharpe：(\frac{\mathbb{E}[r]}{\sigma[r] + \varepsilon})（可设无风险利率=0）
  * 最大回撤：扫描峰值/回撤

**Claude Prompt（评估脚本）**

> 在 `qtmrl/eval/metrics.py` 实现四指标；
> 在 `qtmrl/eval/backtest.py` 实现 `run_backtest(env, actor, deterministic)`，返回指标与 PV/ret 序列；
> 在 `qtmrl/eval/plots.py` 画净值与回撤曲线；
> 在 `scripts/evaluate.py` 加载最优权重在 test split 回测，并输出/保存指标与图表。

---

# 7) 消融与实验矩阵

**目的**：验证论文主张与工程可行性。

**建议矩阵**

* W ∈ {20, 30, 60}
* 指标集：OHLCV-only vs 全指标
* Encoder：TimeCNN vs Transformer
* 奖励：raw return vs log return
* 动作规则：固定 20%/50% vs 10%/30%（稳健性）
* 无 entropy vs entropy=0.05

**Claude Prompt（消融脚本）**

> 在 `scripts/ablation.py` 实现网格实验：读取一组配置变体，循环训练并在验证集评估，保存 CSV 对比表与可视化。

---

# 8) 复现实验的“验收标准”（你可据此验收 Claude 产物）

* 代码能从零生成 `processed` 数据集（日志显示特征维数 F、资产 N、窗口 W）
* `train.py` 能跑通至少一次完整训练（可缩小 total_env_steps 验收），loss 有下降，entropy 非零
* `evaluate.py` 能输出四指标，并画出净值曲线（PV 单调变化合理，不爆 NaN）
* 修改 `configs/default.yaml` 即可重现不同设定（W、指标集、encoder）
* 固定 `seed` 重跑，指标方差在可接受范围（打印 mean/std）

---

# 9) 风险与缓解

* **动作冲突/现金不足**：实现“best-effort 执行”，逐股检查资金，不足则跳过
* **数值不稳**：奖励缩放、梯度裁剪、value loss 权重、entropy 抑制过拟合
* **数据泄漏**：严格按 train split 拟合 (\mu,\sigma)，不要在 test 上拟合
* **指标缺失**：统一在起始 W+max(lag) 之前丢弃
* **联合动作误解**：确保使用 factorized heads，logprob 为各头相加

---

# 10) 可选强化（第二阶段）

* **连续动作版本（资产权重分配）**：动作改为 (w\in\Delta^{N})，约束 (\sum w_i \le 1)，算法用 PPO/SAC；
* **风险敏感奖励**：Sharpe-aware、Drawdown penalty、CVaR；
* **真实交易摩擦**：滑点、最小成交量、停牌/涨跌停；
* **模型升级**：TimeFormer/Per-asset Transformer + Cross-asset Attention；
* **早停与选择**：在 valid split 上挑最优 checkpoint

---

## ✅ 直接可粘贴给 Claude 的“起步 Prompt”（一步生成核心代码）

> 我需要你基于下面的规范创建一个可运行的 Python 项目 `qtmrl`，目录结构、配置、数据处理、环境、模型（TimeCNN + multi-head actor / critic）、A2C 算法、训练与评估脚本，全部按我提供的规范实现：
>
> 1. 创建目录结构与 `configs/default.yaml`（我上面提供的 YAML 原样使用）；
> 2. 在 `scripts/preprocess.py`：读取日K（可先用模拟 CSV），计算指标（`qtmrl/indicators.py`），按训练集做 z-score，生成 `data/processed/` 数据与 `splits.json`；
> 3. 在 `qtmrl/env.py`：实现共享资金池的多资产环境（买20%、卖50%、费率0.05%、禁止做空），状态含 `X_window`、`pos`、`cash`；
> 4. 在 `qtmrl/models/`：实现 `TimeCNNEncoder`（Conv1d→pool），`MultiHeadActor`（N×3 logits），`Critic`（scalar V(s)），支持共享 encoder；
> 5. 在 `qtmrl/algo/`：实现 A2C（rollout=50，gamma=0.96，entropy=0.05，value_coef=1.0，grad_clip=1.0）；
> 6. 在 `scripts/train.py`：加载配置→构建 env/model/trainer→训练到 `total_env_steps`，日志记录 loss/entropy/value_loss/avg_reward；
> 7. 在 `qtmrl/eval/` 与 `scripts/evaluate.py`：按 test split 回测，输出总收益/Sharpe/波动/MDD，保存净值曲线；
> 8. 所有模块加上形状断言与最小单测（尤其 env 的买卖/费用/PV 逻辑）。
>    输出完整代码，并给出运行命令（preprocess→train→evaluate）。

---
