"""Minimal validation script for Phase 0 (works without all dependencies)

This script validates what it can with available packages:
1. Import validation (graceful failures)
2. Basic data structures
3. Environment logic
4. Model architectures (if torch available)

Usage:
    python scripts/quick_validation_minimal.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


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
        "warning": "⚠",
        "skip": "○"
    }
    icon = icons.get(status, "→")
    print(f"{icon} {message}")


def validate_imports():
    """Step 1: Validate module imports"""
    print_header("Step 1: Validating Module Imports")

    results = {}

    # Test each module individually
    modules_to_test = [
        ("qtmrl.utils", ["set_seed", "Config", "load_config", "Logger"]),
        ("qtmrl.dataset", ["StockDataset", "reshape_to_tensor"]),
        ("qtmrl.indicators", ["calculate_all_indicators"]),
        ("qtmrl.env", ["TradingEnv", "Action"]),
        ("qtmrl.models", ["TimeCNNEncoder", "TransformerEncoder", "MultiHeadActor", "Critic"]),
        ("qtmrl.algo", ["RolloutBuffer", "A2CTrainer"]),
        ("qtmrl.eval", ["calculate_all_metrics", "run_backtest"]),
    ]

    for module_name, components in modules_to_test:
        try:
            module = __import__(module_name, fromlist=components)
            for comp in components:
                getattr(module, comp)
            print_status(f"{module_name} → OK", "success")
            results[module_name] = True
        except Exception as e:
            print_status(f"{module_name} → FAILED: {e}", "error")
            results[module_name] = False

    all_passed = all(results.values())
    if all_passed:
        print_status("All modules imported successfully", "success")
    else:
        failed = [k for k, v in results.items() if not v]
        print_status(f"Failed modules: {', '.join(failed)}", "error")

    return results


def validate_data_structures():
    """Step 2: Validate basic data structures without external downloads"""
    print_header("Step 2: Validating Data Structures")

    try:
        import numpy as np
        import pandas as pd

        print_status("Creating sample data...")

        # Create synthetic stock data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        assets = ['STOCK_A', 'STOCK_B']

        data_rows = []
        for asset in assets:
            for date in dates:
                data_rows.append({
                    'date': date,
                    'asset': asset,
                    'Open': 100 + np.random.randn(),
                    'High': 102 + np.random.randn(),
                    'Low': 98 + np.random.randn(),
                    'Close': 100 + np.random.randn(),
                    'Volume': 1000000 + np.random.randint(-100000, 100000)
                })

        df = pd.DataFrame(data_rows)
        print_status(f"Created synthetic data: {len(df)} rows, {len(assets)} assets", "success")

        # Test data reshaping
        from qtmrl.dataset import reshape_to_tensor

        # Create simple feature array
        features = df[['Open', 'High', 'Low', 'Close']].values
        features = features.reshape(len(dates), len(assets), 4)

        window = 10
        states = reshape_to_tensor(features, window)
        print_status(f"Reshaped to tensor: {states.shape}", "success")

        print_status("Data structure validation passed", "success")
        return True

    except Exception as e:
        print_status(f"Data structure validation failed: {e}", "error")
        import traceback
        traceback.print_exc()
        return False


def validate_environment():
    """Step 3: Validate trading environment"""
    print_header("Step 3: Validating Trading Environment")

    try:
        import numpy as np
        from qtmrl.env import TradingEnv, Action

        print_status("Creating test environment...")

        # Create minimal test data
        n_steps = 50
        window = 5
        n_assets = 2
        n_features = 4

        states = np.random.randn(n_steps, window, n_assets, n_features)
        price_changes = np.random.randn(n_steps, n_assets) * 0.01

        env = TradingEnv(
            states=states,
            price_changes=price_changes,
            initial_capital=10000.0,
            fee_rate=0.0005,
            buy_pct=0.2,
            sell_pct=0.5
        )

        print_status(f"Environment created: {n_assets} assets, {n_steps} steps", "success")

        # Test reset
        state = env.reset()
        print_status(f"Reset successful: state shape = {state.shape}", "success")

        # Test actions
        print_status("Testing BUY/HOLD/SELL actions...")

        test_cases = [
            ([Action.BUY, Action.BUY], "BUY for both assets"),
            ([Action.HOLD, Action.HOLD], "HOLD for both assets"),
            ([Action.SELL, Action.SELL], "SELL for both assets"),
            ([Action.BUY, Action.SELL], "Mixed actions"),
        ]

        for actions, description in test_cases:
            state, reward, done, info = env.step(np.array(actions))
            print_status(f"  {description}: reward={reward:.4f}", "info")
            if done:
                print_status("  Episode ended", "info")
                break

        print_status("Environment validation passed", "success")
        return True

    except Exception as e:
        print_status(f"Environment validation failed: {e}", "error")
        import traceback
        traceback.print_exc()
        return False


def validate_models():
    """Step 4: Validate model architectures"""
    print_header("Step 4: Validating Model Architectures")

    try:
        import torch
        print_status("PyTorch available", "success")
    except ImportError:
        print_status("PyTorch not available, skipping model validation", "skip")
        return None

    try:
        from qtmrl.models import create_models
        import torch

        print_status("Creating models...")

        n_features = 10
        n_assets = 2
        d_model = 32
        n_heads = 2
        n_layers = 1

        actor, critic = create_models(
            n_features=n_features,
            n_assets=n_assets,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=0.1
        )

        print_status("Models created successfully", "success")

        # Test forward pass
        print_status("Testing forward pass...")

        batch_size = 4
        window = 5
        dummy_state = torch.randn(batch_size, window, n_assets, n_features)

        with torch.no_grad():
            action_probs = actor(dummy_state)
            value = critic(dummy_state)

        print_status(f"  Actor output: {len(action_probs)} heads, shape={action_probs[0].shape}", "info")
        print_status(f"  Critic output: shape={value.shape}", "info")

        print_status("Model validation passed", "success")
        return True

    except Exception as e:
        print_status(f"Model validation failed: {e}", "error")
        import traceback
        traceback.print_exc()
        return False


def validate_metrics():
    """Step 5: Validate evaluation metrics"""
    print_header("Step 5: Validating Evaluation Metrics")

    try:
        import numpy as np
        from qtmrl.env import TradingEnv
        from qtmrl.eval import calculate_all_metrics

        print_status("Creating test environment...")

        # Simple test environment
        n_steps = 100
        states = np.random.randn(n_steps, 5, 2, 4)
        price_changes = np.random.randn(n_steps, 2) * 0.01 + 0.001

        env = TradingEnv(
            states=states,
            price_changes=price_changes,
            initial_capital=10000.0
        )

        # Run random policy
        print_status("Running random policy...")
        state = env.reset()
        done = False

        while not done:
            actions = np.random.randint(0, 3, size=2)
            state, reward, done, info = env.step(actions)

        # Calculate metrics
        print_status("Calculating metrics...")
        metrics = calculate_all_metrics(env, annualized=True)

        print_status("Metrics calculated:", "info")
        print_status(f"  Total Return: {metrics['total_return']:.2%}", "info")
        print_status(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}", "info")
        print_status(f"  Max Drawdown: {metrics['max_drawdown']:.2%}", "info")
        print_status(f"  Volatility: {metrics['volatility']:.2%}", "info")

        print_status("Metrics validation passed", "success")
        return True

    except Exception as e:
        print_status(f"Metrics validation failed: {e}", "error")
        import traceback
        traceback.print_exc()
        return False


def generate_report(results):
    """Generate validation report"""
    print_header("Validation Summary")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nTimestamp: {timestamp}")
    print(f"Python: {sys.version.split()[0]}")

    # Check package versions
    packages = {}
    for pkg_name in ['numpy', 'pandas', 'torch', 'yaml']:
        try:
            if pkg_name == 'yaml':
                import yaml
                packages['pyyaml'] = 'installed'
            else:
                pkg = __import__(pkg_name)
                packages[pkg_name] = pkg.__version__
        except ImportError:
            packages[pkg_name] = 'NOT INSTALLED'

    print("\nPackages:")
    for pkg, version in packages.items():
        print(f"  {pkg}: {version}")

    print("\nValidation Results:")
    print("-" * 60)

    passed_count = 0
    failed_count = 0
    skipped_count = 0

    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
            passed_count += 1
        elif result is False:
            status = "✗ FAIL"
            failed_count += 1
        else:
            status = "○ SKIP"
            skipped_count += 1

        print(f"{status:10} {test_name}")

    print("-" * 60)
    print(f"Passed: {passed_count}, Failed: {failed_count}, Skipped: {skipped_count}")

    if failed_count == 0:
        print("\n✓ All available tests passed!")
        if skipped_count > 0:
            print(f"  ({skipped_count} test(s) skipped due to missing dependencies)")
        print("\n→ System validation successful")
        success = True
    else:
        print(f"\n✗ {failed_count} test(s) failed")
        print("→ Please fix the issues before proceeding")
        success = False

    # Save report
    report_path = Path("results/validation_report_minimal.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        f.write(f"QTMRL Minimal Validation Report\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Packages:\n")
        for pkg, version in packages.items():
            f.write(f"  {pkg}: {version}\n")
        f.write("\n")

        f.write("Results:\n")
        f.write("-" * 60 + "\n")
        for test_name, result in results.items():
            status = "PASS" if result is True else "FAIL" if result is False else "SKIP"
            f.write(f"  {status:6} {test_name}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Passed: {passed_count}, Failed: {failed_count}, Skipped: {skipped_count}\n")

    print(f"\nReport saved to: {report_path}\n")
    print("=" * 60)

    return success


def main():
    """Main validation workflow"""
    print("\n" + "=" * 60)
    print("  QTMRL Phase 0: Minimal Validation")
    print("  Testing with available dependencies")
    print("=" * 60)

    results = {}

    # Run tests
    import_results = validate_imports()
    results["1. Module Imports"] = all(import_results.values())

    if import_results.get("qtmrl.dataset", False):
        results["2. Data Structures"] = validate_data_structures()
    else:
        results["2. Data Structures"] = None

    if import_results.get("qtmrl.env", False):
        results["3. Trading Environment"] = validate_environment()
    else:
        results["3. Trading Environment"] = None

    if import_results.get("qtmrl.models", False):
        results["4. Model Architectures"] = validate_models()
    else:
        results["4. Model Architectures"] = None

    if import_results.get("qtmrl.eval", False):
        results["5. Evaluation Metrics"] = validate_metrics()
    else:
        results["5. Evaluation Metrics"] = None

    # Generate report
    success = generate_report(results)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
