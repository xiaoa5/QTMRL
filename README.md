# QTMRL - Quantitative Trading with Multi-asset Reinforcement Learning

> **⚠️ Note**: This is an **unofficial implementation** of the research paper. This project is not affiliated with the original authors.

**Reference Paper**:
> **Title**: QTMRL: Quantitative Trading with Multi-asset Reinforcement Learning
> **Authors**: Xiangdong Liu, Jiahao Chen
> **Link**: [https://arxiv.org/pdf/2508.20467v1]

QTMRL is a multi-asset quantitative trading reinforcement learning system based on the **A2C (Advantage Actor-Critic)** algorithm. It uses daily OHLCV data and technical indicators to learn multi-asset trading strategies via a factorized multi-head policy, supporting shared capital pools and portfolio-level rewards.

## 🌟 Features

- ✅ **Fully Reproducible**: Fixed random seeds, automated data download, one-click execution.
- 📊 **Multi-Asset Trading**: Supports simultaneous trading of multiple stocks with a shared capital pool.
- 🧠 **A2C Algorithm**: Policy gradient method based on Advantage Actor-Critic.
- 📈 **Rich Technical Indicators**: SMA, EMA, RSI, MACD, ATR, Bollinger Bands, Ichimoku, SuperTrend, etc.
- 🔧 **Flexible Configuration**: YAML configuration files for easy parameter tuning.
- 📉 **Comprehensive Evaluation**: Returns, Sharpe Ratio, Volatility, Max Drawdown, etc.
- 🎨 **Visualization**: Portfolio value curves, drawdown curves, return distributions, action distributions.
- 🚀 **Colab Support**: Optimized for Google Colab environment with GPU support.
- 📝 **Wandb Integration**: Supports experiment tracking and visualization.

---

## 🚀 Quick Start

### 1. Requirements

- Python 3.8+
- CUDA (Optional, for GPU acceleration)

### 2. Installation

#### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/QTMRL.git
cd QTMRL

# Install dependencies
pip install -r requirements.txt

# Or install in editable mode
pip install -e .
```

#### Google Colab Installation

```python
# Run in a Colab notebook cell
!git clone https://github.com/yourusername/QTMRL.git
%cd QTMRL
!pip install -r requirements.txt
```

### 3. Run the Full Pipeline

#### Step 1: Data Preprocessing

Download stock data and calculate technical indicators:

```bash
# Use quick test configuration (4 stocks, 2022-2024) - Recommended for first run
python scripts/preprocess.py --config configs/quick_test.yaml

# Or use default configuration (16 stocks, 2014-2024)
python scripts/preprocess.py --config configs/default.yaml
```

Processed data will be saved in the `data/processed/` directory.

#### Step 2: Validation (Optional but Recommended)

Verify that the training pipeline works correctly before starting a long training session:

```bash
python scripts/quick_validation.py
```

#### Step 3: Train Model

```bash
# Quick Test (50K steps, ~10-20 mins)
python scripts/train.py --config configs/quick_test.yaml

# Full Training (1M steps, ~2-3 hours)
python scripts/train.py --config configs/default.yaml
```

During training, the system will:
- Automatically save checkpoints to `checkpoints/` or `runs/`.
- Periodically evaluate on the validation set.
- Log training metrics (loss, entropy, reward, etc.).

#### Step 4: Evaluate Model

You can evaluate a trained model or test the pipeline with a random policy.

**Evaluate Trained Model:**

```bash
python scripts/evaluate.py \
    --config configs/quick_test.yaml \
    --model runs/final_model.pth \
    --split test \
    --save-plots
```

**Evaluate Random Policy (No Model Required):**

```bash
# Useful for testing the evaluation pipeline without training
python scripts/evaluate.py --config configs/quick_test.yaml --save-plots
```

Evaluation results include:
- Total Return, Annualized Return
- Sharpe Ratio, Annualized Sharpe Ratio
- Volatility, Annualized Volatility
- Max Drawdown
- Visualization plots in `results/` directory

---

## ⚙️ Configuration

### Default Configuration (`configs/default.yaml`)

```yaml
# Asset List (16 US Stocks)
assets: [AAPL, MSFT, NVDA, CVX, OXY, AAL, UAL, DAL, CCL, RCL, WYNN, LVS, AXP, BAC, JNJ, GOOGL]

# Data Split (2014-2024)
split:
  train: ["2014-01-02", "2019-12-31"]  # 6 years
  valid: ["2020-01-02", "2022-12-31"]  # 3 years
  test:  ["2023-01-02", "2024-12-31"]  # 2 years

# Trading Parameters
window: 20              # State window length
fee_rate: 0.0005       # Transaction fee rate 0.05%
buy_pct: 0.20          # Use 20% of cash for buying
sell_pct: 0.50         # Sell 50% of position
initial_cash: 100000   # Initial cash $100,000

# Model Parameters
model:
  encoder: "TimeCNN"   # Encoder type: TimeCNN or Transformer
  d_model: 128         # Model dimension
  n_layers: 3          # Number of layers

# Training Parameters
train:
  total_env_steps: 1000000  # Total steps
  rollout_steps: 50         # Rollout steps
  gamma: 0.96               # Discount factor
  entropy_coef: 0.05        # Entropy coefficient
  lr_actor: 1.0e-5          # Actor learning rate
  lr_critic: 1.0e-5         # Critic learning rate
```

### Quick Test Configuration (`configs/quick_test.yaml`)

Designed for quick verification:
- 4 Stocks: AAPL, MSFT, NVDA, GOOGL
- Data Range: 2022-2024
- Training Steps: 50K
- Smaller Model: d_model=64, n_layers=2

---

## 🛠 Troubleshooting & Recent Fixes

### Common Issues

**Q: Download failed?**
A: Check your internet connection. `yfinance` will automatically retry.

**Q: Out of memory?**
A: Use `quick_test.yaml` or reduce the number of assets and `d_model`.

**Q: "Unexpected keyword argument 'window_size'" error?**
A: This has been fixed in the latest version. Ensure you are using the updated `qtmrl/models/encoders.py` and `qtmrl/models/actor_critic.py`.

### Recent Improvements

1.  **Robust Window Handling**: Fixed issues with small window sizes causing convolution errors. The system now dynamically adjusts kernel sizes and uses padding to support any window size (W >= 1).
2.  **Numerical Stability**: Added Xavier initialization and NaN checks to prevent training instability.
3.  **Evaluation Updates**: `evaluate.py` now supports random policy evaluation (if no model is provided) and correctly handles model loading with dynamic layer initialization.
4.  **Data Consistency**: Fixed rollout buffer errors caused by variable window sizes at the beginning of episodes.

---

## 📂 Project Structure

```
QTMRL/
├── configs/                    # Configuration files
│   ├── default.yaml           # Default config
│   └── quick_test.yaml        # Quick test config
├── data/                       # Data directory
│   ├── raw/                   # Raw downloaded data
│   └── processed/             # Preprocessed numpy data
├── qtmrl/                      # Core package
│   ├── indicators.py          # Technical indicators
│   ├── dataset.py             # Dataset management
│   ├── env.py                 # Trading environment
│   ├── models/                # Model definitions (Encoders, Actor-Critic)
│   ├── algo/                  # RL Algorithms (A2C, RolloutBuffer)
│   ├── eval/                  # Evaluation modules (Backtest, Metrics, Plots)
│   └── utils/                 # Utilities (Seed, Config, IO)
├── scripts/                    # Executable scripts
│   ├── preprocess.py          # Data preprocessing
│   ├── train.py               # Training loop
│   ├── evaluate.py            # Evaluation & Visualization
│   └── quick_validation.py    # System validation script
├── tests/                      # Unit tests
├── README.md                   # Original Chinese README
├── USER_GUIDE_EN.md            # This English Guide
├── requirements.txt            # Dependencies
└── setup.py                    # Setup script
```

---

## 💡 Motivation & Acknowledgements

I was intrigued by the remarkable results presented in the original paper and wanted to reproduce them. However, since the GitHub link provided in the paper was invalid, I decided to implement the system from scratch.

Special thanks to **Claude Code** and **Antigravity** for their powerful assistance, which enabled me to complete this full reproduction in just **two nights**!

## ⚖️ Disclaimer

This project is for research and educational purposes only. It does not constitute investment advice. Use this system for actual trading at your own risk.
