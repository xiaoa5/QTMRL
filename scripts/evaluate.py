"""评估脚本"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch

from qtmrl.utils import load_config, set_seed, load_json
from qtmrl.env import TradingEnv
from qtmrl.models import create_models
from qtmrl.eval import (
    run_backtest,
    print_metrics,
    plot_portfolio_value,
    plot_drawdown,
    plot_returns_distribution,
    plot_action_distribution,
)


def main():
    parser = argparse.ArgumentParser(description="评估训练好的模型")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型checkpoint路径",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="评估的数据集分割",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="保存图表",
    )
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    print(f"加载配置: {args.config}")

    # 设置随机种子
    set_seed(config.seed)

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # ========== 1. 加载数据 ==========
    print("\n" + "=" * 60)
    print(f"加载 {args.split} 数据")
    print("=" * 60)

    metadata = load_json("data/processed/metadata.json")
    n_assets = metadata["n_assets"]
    n_features = metadata["n_features"]
    asset_names = metadata["assets"]

    X = np.load(f"data/processed/X_{args.split}.npy")
    Close = np.load(f"data/processed/Close_{args.split}.npy")
    dates = np.load(f"data/processed/dates_{args.split}.npy", allow_pickle=True)

    print(f"数据形状: {X.shape}")
    print(f"日期范围: {dates[0]} 到 {dates[-1]}")

    # ========== 2. 创建环境 ==========
    print("\n创建交易环境")

    env = TradingEnv(
        X=X,
        Close=Close,
        dates=dates,
        window=config.window,
        initial_cash=config.initial_cash,
        fee_rate=config.fee_rate,
        buy_pct=config.buy_pct,
        sell_pct=config.sell_pct,
    )

    # ========== 3. 加载模型 ==========
    print("\n加载模型")

    actor, critic = create_models(config, n_assets, n_features)

    checkpoint = torch.load(args.model, map_location=device)
    actor.load_state_dict(checkpoint["actor"])
    actor.to(device)

    print(f"模型已加载: {args.model}")

    # ========== 4. 运行回测 ==========
    print("\n" + "=" * 60)
    print("运行回测")
    print("=" * 60)

    metrics, portfolio_values, actions_history = run_backtest(
        env, actor, device=device, deterministic=True
    )

    # 打印指标
    print_metrics(metrics)

    # 计算收益率
    returns = env.get_returns()

    # ========== 5. 可视化 ==========
    print("\n" + "=" * 60)
    print("生成可视化")
    print("=" * 60)

    output_dir = Path("results") / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    # 组合净值曲线
    plot_portfolio_value(
        portfolio_values,
        dates=dates[: len(portfolio_values)],
        title=f"组合净值曲线 ({args.split})",
        save_path=str(output_dir / "portfolio_value.png")
        if args.save_plots
        else None,
    )

    # 回撤曲线
    plot_drawdown(
        portfolio_values,
        dates=dates[: len(portfolio_values)],
        title=f"回撤曲线 ({args.split})",
        save_path=str(output_dir / "drawdown.png") if args.save_plots else None,
    )

    # 收益率分布
    plot_returns_distribution(
        returns,
        title=f"收益率分布 ({args.split})",
        save_path=str(output_dir / "returns_dist.png")
        if args.save_plots
        else None,
    )

    # 动作分布
    if len(asset_names) <= 8:  # 只在资产不太多时绘制
        plot_action_distribution(
            actions_history,
            asset_names=asset_names,
            title=f"动作分布 ({args.split})",
            save_path=str(output_dir / "actions_dist.png")
            if args.save_plots
            else None,
        )

    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)

    if args.save_plots:
        print(f"\n图表已保存到: {output_dir}")


if __name__ == "__main__":
    main()
