"""Quick validation script for Phase 0

This script validates that the current QTMRL system works end-to-end:
1. Data preprocessing pipeline
2. Model training (minimal steps)
3. Evaluation and metrics
4. Generate validation report

Usage:
    python scripts/quick_validation.py
"""

import sys
import os
from pathlib import Path
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from qtmrl.utils import set_seed, load_config, Logger
from qtmrl.dataset import StockDataset
from qtmrl.indicators import calculate_all_indicators
from qtmrl.env import TradingEnv
from qtmrl.models import create_models
from qtmrl.algo import A2CTrainer
from qtmrl.eval import calculate_all_metrics, run_backtest


def print_header(title):
    """Print formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_status(message, status="info"):
    """Print status message with icon"""
    icons = {
        "info": "→",
        "success": "✓",
        "error": "✗",
        "warning": "⚠"
    }
    icon = icons.get(status, "→")
    print(f"{icon} {message}")


def validate_imports():
    """Step 1: Validate all modules can be imported"""
    print_header("Step 1: Validating Imports")

    try:
        from qtmrl.utils import set_seed, Config, load_config, Logger
        from qtmrl.dataset import StockDataset, reshape_to_tensor
        from qtmrl.indicators import calculate_all_indicators
        from qtmrl.env import TradingEnv, Action
        from qtmrl.models import TimeCNNEncoder, TransformerEncoder, MultiHeadActor, Critic
        from qtmrl.algo import RolloutBuffer, A2CTrainer
        from qtmrl.eval import calculate_all_metrics, run_backtest

        print_status("All core modules imported successfully", "success")
        return True
    except Exception as e:
        print_status(f"Import failed: {e}", "error")
        traceback.print_exc()
        return False


def validate_data_preprocessing():
    """Step 2: Validate data preprocessing pipeline"""
    print_header("Step 2: Validating Data Preprocessing")

    try:
        # Use minimal config for fast testing
        print_status("Creating test dataset (2 stocks, 2023-2024)...")

        dataset = StockDataset(
            assets=["AAPL", "MSFT"],
            start_date="2023-01-01",
            end_date="2024-01-01",
            data_dir="data/cache"
        )

        print_status("Downloading data from yfinance...")
        data_dict = dataset.download_data()
        df = dataset.align_data()

        if df is None or len(df) == 0:
            print_status("Data download failed", "error")
            return False

        print_status(f"Data loaded: {len(df)} rows", "success")

        # Calculate indicators
        print_status("Calculating technical indicators...")
        df_with_indicators = calculate_all_indicators(df)

        if df_with_indicators is None or len(df_with_indicators) == 0:
            print_status("Indicator calculation failed", "error")
            return False

        n_indicators = len([col for col in df_with_indicators.columns
                           if col not in ['date', 'asset', 'Open', 'High', 'Low', 'Close', 'Volume']])
        print_status(f"Calculated {n_indicators} indicators", "success")

        # Test data splitting
        train_df, val_df, test_df = dataset.split_data(df_with_indicators, train_ratio=0.6, val_ratio=0.2)
        print_status(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}", "success")

        print_status("Data preprocessing validation passed", "success")
        return True

    except Exception as e:
        print_status(f"Data preprocessing failed: {e}", "error")
        traceback.print_exc()
        return False


def validate_training():
    """Step 3: Validate basic training functionality"""
    print_header("Step 3: Validating Training Pipeline")

    try:
        set_seed(42)

        # Minimal config for quick test
        print_status("Setting up minimal training configuration...")

        # Load quick test config if exists, otherwise use minimal settings
        config_path = Path("configs/quick_test.yaml")
        if config_path.exists():
            config = load_config(config_path)
            # Override for even faster validation
            config.train.total_steps = 100
            config.train.n_envs = 1
        else:
            # Create minimal config programmatically
            class MinimalConfig:
                def __init__(self):
                    self.data = type('obj', (object,), {
                        'assets': ['AAPL', 'MSFT'],
                        'start_date': '2023-01-01',
                        'end_date': '2024-01-01',
                        'window': 10,
                        'indicators': ['sma', 'ema', 'rsi']
                    })
                    self.env = type('obj', (object,), {
                        'initial_capital': 10000.0,
                        'fee_rate': 0.0005,
                        'buy_pct': 0.2,
                        'sell_pct': 0.5
                    })
                    self.model = type('obj', (object,), {
                        'd_model': 64,
                        'n_heads': 2,
                        'n_layers': 2,
                        'dropout': 0.1
                    })
                    self.train = type('obj', (object,), {
                        'total_steps': 100,
                        'n_envs': 1,
                        'n_steps': 128,
                        'learning_rate': 0.0003,
                        'gamma': 0.99,
                        'ent_coef': 0.01,
                        'vf_coef': 0.5,
                        'max_grad_norm': 0.5
                    })

            config = MinimalConfig()

        # Create dataset (reuse from previous test if possible)
        print_status("Preparing training data...")
        dataset = StockDataset(
            assets=config.data.assets[:2],  # Use only 2 assets
            start_date=config.data.start_date,
            end_date=config.data.end_date,
            data_dir="data/cache"
        )

        data_dict = dataset.download_data()
        df = dataset.align_data()
        df = calculate_all_indicators(df)

        # Split data
        total_len = len(df)
        train_end = int(total_len * 0.8)
        train_df = df.iloc[:train_end]

        # Prepare arrays from script/preprocess.py logic
        # Extract features and prices
        feature_cols = [col for col in train_df.columns if col not in ['date', 'asset', 'Open', 'High', 'Low', 'Close', 'Volume']]
        train_features = train_df[feature_cols].values
        train_close = train_df['Close'].values
        train_dates = train_df['date'].values

        # Reshape to [T, N, F]
        n_assets = len(config.data.assets[:2])
        T = len(train_dates) // n_assets
        train_X = train_features.reshape(T, n_assets, -1)
        train_Close = train_close.reshape(T, n_assets)
        train_dates_arr = train_dates[::n_assets]

        print_status(f"Training data: X={train_X.shape}, Close={train_Close.shape}", "success")

        # Create environment
        print_status("Creating training environment...")
        env = TradingEnv(
            X=train_X,
            Close=train_Close,
            dates=train_dates_arr,
            window=config.data.window,
            initial_cash=config.env.initial_capital,
            fee_rate=config.env.fee_rate,
            buy_pct=config.env.buy_pct,
            sell_pct=config.env.sell_pct
        )

        # Create models
        print_status("Creating models...")
        n_features = train_X.shape[-1]
        n_assets = train_X.shape[1]

        actor, critic = create_models(
            n_features=n_features,
            n_assets=n_assets,
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            n_layers=config.model.n_layers,
            dropout=config.model.dropout
        )

        # Create trainer
        print_status("Initializing A2C trainer...")
        trainer = A2CTrainer(
            env=env,
            actor=actor,
            critic=critic,
            learning_rate=config.train.learning_rate,
            gamma=config.train.gamma,
            ent_coef=config.train.ent_coef,
            vf_coef=config.train.vf_coef,
            max_grad_norm=config.train.max_grad_norm,
            n_steps=min(config.train.n_steps, 128)  # Use smaller rollout for speed
        )

        # Run training for minimal steps
        print_status(f"Running training for {config.train.total_steps} steps...")

        for step in range(config.train.total_steps):
            # Collect rollout
            rollout = trainer.collect_rollout()

            # Update policy
            stats = trainer.update(rollout)

            if (step + 1) % 50 == 0:
                print_status(f"  Step {step+1}/{config.train.total_steps}: "
                           f"loss={stats['loss']:.3f}, "
                           f"value_loss={stats['value_loss']:.3f}", "info")

        print_status("Training completed successfully", "success")

        # Test model inference
        print_status("Testing model inference...")
        state = env.reset()
        with torch.no_grad():
            features = torch.FloatTensor(state['features']).unsqueeze(0)  # [1, W, N, F]
            positions = torch.FloatTensor(state['positions']).unsqueeze(0)  # [1, N]
            cash = torch.FloatTensor(state['cash']).unsqueeze(0)  # [1, 1]

            logits, action_probs = actor(features, positions, cash)
            value = critic(features, positions, cash)

        print_status(f"  Action logits shape: {logits.shape}", "info")
        print_status(f"  Action probs shape: {action_probs.shape}", "info")
        print_status(f"  Value shape: {value.shape}", "info")

        print_status("Training validation passed", "success")
        return True

    except Exception as e:
        print_status(f"Training validation failed: {e}", "error")
        traceback.print_exc()
        return False


def validate_evaluation():
    """Step 4: Validate evaluation functionality"""
    print_header("Step 4: Validating Evaluation Pipeline")

    try:
        set_seed(42)

        # Create simple test environment
        print_status("Creating test environment...")

        # Create dummy data for quick testing
        T = 100
        window = 10
        n_assets = 2
        n_features = 10

        X = np.random.randn(T, n_assets, n_features).astype(np.float32)
        Close = np.random.rand(T, n_assets).astype(np.float32) * 100 + 50  # Prices around 50-150
        dates = np.arange(T)

        env = TradingEnv(
            X=X,
            Close=Close,
            dates=dates,
            window=window,
            initial_cash=10000.0,
            fee_rate=0.0005,
            buy_pct=0.2,
            sell_pct=0.5
        )

        # Create simple random policy for testing
        print_status("Running backtest with random policy...")

        state = env.reset()
        done = False

        while not done:
            # Random actions
            actions = np.random.randint(0, 3, size=n_assets)
            state, reward, done, info = env.step(actions)

        # Calculate metrics
        print_status("Calculating evaluation metrics...")
        metrics = calculate_all_metrics(env, annualized=True)

        print_status(f"  Total Return: {metrics['total_return']:.2%}", "info")
        print_status(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}", "info")
        print_status(f"  Max Drawdown: {metrics['max_drawdown']:.2%}", "info")
        print_status(f"  Volatility: {metrics['volatility']:.2%}", "info")

        print_status("Evaluation validation passed", "success")
        return True

    except Exception as e:
        print_status(f"Evaluation validation failed: {e}", "error")
        traceback.print_exc()
        return False


def generate_validation_report(results):
    """Generate validation report"""
    print_header("Validation Report")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\nValidation Time: {timestamp}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"NumPy Version: {np.__version__}")

    print("\nValidation Results:")
    print("-" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10} {test_name}")
        if not passed:
            all_passed = False

    print("-" * 60)

    if all_passed:
        print("\n✓ All validation tests passed!")
        print("\nNext steps:")
        print("  1. Run full preprocessing: python scripts/preprocess.py --config configs/quick_test.yaml")
        print("  2. Run training: python scripts/train.py --config configs/quick_test.yaml")
        print("  3. Run evaluation: python scripts/evaluate.py --config configs/quick_test.yaml")
        print("\n→ System is ready for paper reproduction experiments")
    else:
        print("\n✗ Some validation tests failed")
        print("→ Please fix the issues before proceeding")

    print("\n" + "=" * 60)

    # Save report to file
    report_path = Path("results/validation_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        f.write(f"QTMRL Quick Validation Report\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("=" * 60 + "\n\n")

        f.write("System Information:\n")
        f.write(f"  Python: {sys.version.split()[0]}\n")
        f.write(f"  PyTorch: {torch.__version__}\n")
        f.write(f"  NumPy: {np.__version__}\n\n")

        f.write("Validation Results:\n")
        f.write("-" * 60 + "\n")
        for test_name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            f.write(f"  {status:6} {test_name}\n")
        f.write("-" * 60 + "\n\n")

        if all_passed:
            f.write("Overall: ALL TESTS PASSED\n")
        else:
            f.write("Overall: SOME TESTS FAILED\n")

    print(f"\nReport saved to: {report_path}")

    return all_passed


def main():
    """Main validation workflow"""
    print("\n" + "=" * 60)
    print("  QTMRL Phase 0: Quick Validation")
    print("  Verifying system end-to-end functionality")
    print("=" * 60)

    results = {}

    # Run validation steps
    results["1. Import Validation"] = validate_imports()

    if results["1. Import Validation"]:
        results["2. Data Preprocessing"] = validate_data_preprocessing()
        results["3. Training Pipeline"] = validate_training()
        results["4. Evaluation Pipeline"] = validate_evaluation()
    else:
        print_status("Skipping further tests due to import failure", "warning")
        results["2. Data Preprocessing"] = False
        results["3. Training Pipeline"] = False
        results["4. Evaluation Pipeline"] = False

    # Generate report
    all_passed = generate_validation_report(results)

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
