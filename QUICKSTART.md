# QTMRL 快速开始指南

这是一个5分钟快速上手指南，帮助你快速验证系统可用性。

## 快速测试（推荐首次使用）

使用 `quick_test.yaml` 配置进行快速验证：

```bash
# 1. 数据预处理 (约1-2分钟)
python scripts/preprocess.py --config configs/quick_test.yaml

# 2. 训练模型 (约10-20分钟，50K步)
python scripts/train.py --config configs/quick_test.yaml

# 3. 评估模型
python scripts/evaluate.py \
    --config configs/quick_test.yaml \
    --model runs/final_model.pth \
    --split test \
    --save-plots
```

快速测试配置特点：
- **4只股票**: AAPL, MSFT, NVDA, GOOGL
- **数据范围**: 2022-2024 (约2.5年)
- **训练步数**: 50K (约10-20分钟)
- **模型规模**: 较小 (d_model=64, n_layers=2)

## 完整训练

验证代码可用后，使用完整配置：

```bash
# 1. 数据预处理 (约2-3分钟)
python scripts/preprocess.py --config configs/default.yaml

# 2. 训练模型 (约2-3小时，1M步)
python scripts/train.py --config configs/default.yaml

# 3. 评估模型
python scripts/evaluate.py \
    --config configs/default.yaml \
    --model runs/final_model.pth \
    --split test \
    --save-plots
```

完整配置特点：
- **16只股票**: 涵盖科技、能源、航空、旅游、金融、医药等行业
- **数据范围**: 2014-2024 (10年)
- **训练步数**: 1M (约2-3小时)
- **模型规模**: 标准 (d_model=128, n_layers=3)

## 在Google Colab上运行

### 快速测试

```python
# 1. 克隆并安装
!git clone https://github.com/xiaoa5/QTMRL.git
%cd QTMRL
!pip install -q -r requirements.txt

# 2. 数据预处理
!python scripts/preprocess.py --config configs/quick_test.yaml

# 3. 训练 (使用GPU)
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
display(Image('results/test/drawdown.png'))
```

### 保存结果到Google Drive

```python
# 挂载Drive
from google.colab import drive
drive.mount('/content/drive')

# 复制结果到Drive
!cp -r runs /content/drive/MyDrive/QTMRL_runs
!cp -r results /content/drive/MyDrive/QTMRL_results

print("结果已保存到Google Drive!")
```

## 使用Wandb跟踪实验

1. 登录Wandb（首次使用）:

```python
import wandb
wandb.login()
```

2. 修改配置文件启用Wandb:

```yaml
logging:
  use_wandb: true
  wandb_project: "qtmrl"
```

3. 运行训练，实验会自动上传到 https://wandb.ai

## 常见问题

### Q: 下载数据失败怎么办？

A: 检查网络连接，或者重新运行preprocess.py，yfinance会自动重试。

### Q: 内存不足怎么办？

A: 使用 `quick_test.yaml` 配置，或者减少资产数量和模型维度。

### Q: 想更换股票池？

A: 修改配置文件中的 `assets` 列表即可。

### Q: 如何调整训练时间？

A: 修改 `total_env_steps`，推荐值：
- 快速测试: 50K
- 初步验证: 100K-200K
- 完整训练: 500K-1M

## 下一步

- 阅读完整 [README.md](README.md)
- 查看 [plan.md](plan.md) 了解系统设计
- 尝试修改配置文件进行实验
- 实现消融实验（不同窗口、指标、模型等）

## 验证安装

运行单元测试确保环境正确：

```bash
python tests/test_env.py
```

应该看到：
```
运行交易环境单元测试...

✓ 环境初始化测试通过
✓ 环境重置测试通过
✓ HOLD动作测试通过
✓ BUY动作测试通过
✓ SELL动作测试通过
✓ 组合价值计算测试通过
✓ 手续费计算测试通过

所有测试通过! ✓
```

## 预期结果

快速测试预期训练指标：
- **Loss**: 应该逐渐下降
- **Entropy**: 保持在0.5-1.5之间
- **Avg Reward**: 可能在-0.01到0.01之间波动
- **Portfolio Value**: 应该有明显变化

注意：由于市场随机性和训练时间较短，快速测试的性能可能不稳定，这是正常的。完整训练会有更好的稳定性。

---

祝你使用愉快！如有问题欢迎提Issue。
