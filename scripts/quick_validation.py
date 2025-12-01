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

from qtmrl.utils import set_seed, load_config, Config, Logger
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
        # Create a simple indicator config
        indicator_config = {
            "sma": [10, 20],
            "ema": [12, 26],
            "rsi": [14]
        }
        df_with_indicators = calculate_all_indicators(df, indicator_config)

        if df_with_indicators is None or len(df_with_indicators) == 0:
            print_status("Indicator calculation failed", "error")
            return False

        n_indicators = len([col for col in df_with_indicators.columns
                           if col not in ['date', 'asset', 'Open', 'High', 'Low', 'Close', 'Volume']])
        print_status(f"Calculated {n_indicators} indicators", "success")

        # Test data splitting (using simple index-based split for validation)
        total_len = len(df_with_indicators)
        train_end = int(total_len * 0.6)
        val_end = int(total_len * 0.8)
        train_df = df_with_indicators.iloc[:train_end]
        val_df = df_with_indicators.iloc[train_end:val_end]
        test_df = df_with_indicators.iloc[val_end:]
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
            # Extract parameters from config structure
            assets = config.assets[:2]  # Top-level assets
            start_date = config.split.train[0]
            end_date = config.split.train[1]
            window = config.window
            initial_cash = config.initial_cash
            fee_rate = config.fee_rate
            buy_pct = config.buy_pct
            sell_pct = config.sell_pct
            indicator_config = config.features.indicators.to_dict()

            # Model params
            d_model = config.model.d_model
            n_layers = config.model.n_layers
            dropout = config.model.dropout
            n_heads = 2  # Not in config, use default

            # Training params
            total_steps = 100  # Override for fast validation
            learning_rate = config.train.lr_actor
            gamma = config.train.gamma
            ent_coef = config.train.entropy_coef
            vf_coef = config.train.value_coef
            max_grad_norm = config.train.grad_clip
            n_steps = config.train.rollout_steps
        else:
            # Create minimal config
            assets = ['AAPL', 'MSFT']
            start_date = '2023-01-01'
            end_date = '2024-01-01'
            window = 10
            initial_cash = 10000.0
            fee_rate = 0.0005
            buy_pct = 0.2
            sell_pct = 0.5
            indicator_config = {
                'sma': [10, 20],
                'ema': [12, 26],
                'rsi': [14]
            }

            d_model = 64
            n_heads = 2
            n_layers = 2
            dropout = 0.1

            total_steps = 100
            learning_rate = 0.0003
            gamma = 0.99
            ent_coef = 0.01
            vf_coef = 0.5
            max_grad_norm = 0.5
            n_steps = 128

        # Create dataset (reuse from previous test if possible)
        print_status("Preparing training data...")
        dataset = StockDataset(
            assets=assets,
            start_date=start_date,
            end_date=end_date,
            data_dir="data/cache"
        )

        data_dict = dataset.download_data()
        df = dataset.align_data()
        df = calculate_all_indicators(df, indicator_config)

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
        n_assets_count = len(assets)
        T = len(train_dates) // n_assets_count
        train_X = train_features.reshape(T, n_assets_count, -1)
        train_Close = train_close.reshape(T, n_assets_count)
        train_dates_arr = train_dates[::n_assets_count]

        print_status(f"Training data: X={train_X.shape}, Close={train_Close.shape}", "success")

        # Create environment
        print_status("Creating training environment...")
        env = TradingEnv(
            X=train_X,
            Close=train_Close,
            dates=train_dates_arr,
            window=window,
            initial_cash=initial_cash,
            fee_rate=fee_rate,
            buy_pct=buy_pct,
            sell_pct=sell_pct
        )

        # Create models
        print_status("Creating models...")
        n_features = train_X.shape[-1]
        n_assets_model = train_X.shape[1]

        # Create a config object for create_models
        if config_path.exists():
            model_config = config
        else:
            # Create minimal config with model parameters
            model_config_dict = {
                'model': {
                    'd_model': d_model,
                    'n_layers': n_layers,
                    'dropout': dropout,
                    'encoder': 'TimeCNN'
                }
            }
            model_config = Config(model_config_dict)

        actor, critic = create_models(
            config=model_config,
            n_assets=n_assets_model,
            n_features=n_features
        )

        # Create trainer
        print_status("Initializing A2C trainer...")
        trainer = A2CTrainer(
            actor=actor,
            critic=critic,
            lr_actor=learning_rate,
            lr_critic=learning_rate,
            gamma=gamma,
            entropy_coef=ent_coef,
            value_coef=vf_coef,
            grad_clip=max_grad_norm
        )

        # Create rollout buffer
        from qtmrl.algo import RolloutBuffer
        buffer = RolloutBuffer()

        # Run training for minimal steps
        print_status(f"Running training for {total_steps} steps...")

        rollout_steps = min(n_steps, 128)
        num_rollouts = total_steps // rollout_steps

        for rollout_idx in range(num_rollouts):
            # Collect rollout
            rollout_stats = trainer.collect_rollout(env, rollout_steps, buffer)

            # Update policy
            update_stats = trainer.update(buffer)

            if (rollout_idx + 1) % 5 == 0:
                steps_done = (rollout_idx + 1) * rollout_steps
                print_status(f"  Step {steps_done}/{total_steps}: "
                           f"avg_reward={rollout_stats.get('avg_reward', 0):.3f}", "info")

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
        portfolio_values = np.array(env.portfolio_values)
        metrics = calculate_all_metrics(portfolio_values, annualize=True)

        print_status(f"  Total Return: {metrics['total_return']:.2%}", "info")
        print_status(f"  Sharpe Ratio: {metrics['sharpe']:.3f}", "info")
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
