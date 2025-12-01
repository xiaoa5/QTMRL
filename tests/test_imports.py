"""测试所有模块导入"""
import sys

def test_imports():
    """测试所有关键模块是否能正常导入"""
    errors = []

    # 测试工具模块
    try:
        from qtmrl.utils import (
            set_seed, Config, load_config, Logger,
            save_json, load_json
        )
        print("✓ qtmrl.utils 导入成功")
    except Exception as e:
        errors.append(f"✗ qtmrl.utils 导入失败: {e}")

    # 测试数据模块
    try:
        from qtmrl.dataset import StockDataset, reshape_to_tensor
        from qtmrl.indicators import calculate_all_indicators
        print("✓ qtmrl.dataset 和 qtmrl.indicators 导入成功")
    except Exception as e:
        errors.append(f"✗ 数据模块导入失败: {e}")

    # 测试环境
    try:
        from qtmrl.env import TradingEnv, Action
        print("✓ qtmrl.env 导入成功")
    except Exception as e:
        errors.append(f"✗ qtmrl.env 导入失败: {e}")

    # 测试模型
    try:
        from qtmrl.models import (
            TimeCNNEncoder, TransformerEncoder,
            MultiHeadActor, Critic, create_models
        )
        print("✓ qtmrl.models 导入成功")
    except Exception as e:
        errors.append(f"✗ qtmrl.models 导入失败: {e}")

    # 测试算法
    try:
        from qtmrl.algo import RolloutBuffer, A2CTrainer
        print("✓ qtmrl.algo 导入成功")
    except Exception as e:
        errors.append(f"✗ qtmrl.algo 导入失败: {e}")

    # 测试评估
    try:
        from qtmrl.eval import (
            calculate_all_metrics, run_backtest,
            plot_portfolio_value
        )
        print("✓ qtmrl.eval 导入成功")
    except Exception as e:
        errors.append(f"✗ qtmrl.eval 导入失败: {e}")

    # 打印错误
    if errors:
        print("\n发现错误:")
        for error in errors:
            print(error)
        return False
    else:
        print("\n所有模块导入成功! ✓")
        return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
