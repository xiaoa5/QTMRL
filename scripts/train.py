"""训练脚本"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
from tqdm import tqdm

from qtmrl.utils import load_config, set_seed, Logger, load_json
from qtmrl.env import TradingEnv
from qtmrl.models import create_models
from qtmrl.algo import A2CTrainer
from qtmrl.eval import run_backtest, print_metrics


def main():
    parser = argparse.ArgumentParser(description="训练A2C交易agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="恢复训练的checkpoint路径",
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

    # 创建输出目录
    output_dir = Path(config.logging["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建日志记录器
    logger = Logger(
        output_dir=str(output_dir),
        use_wandb=config.logging.get("use_wandb", False),
        wandb_project=config.logging.get("wandb_project", "qtmrl"),
        wandb_entity=config.logging.get("wandb_entity", None),
        wandb_name=f"train_{config.seed}",
        wandb_config=config.to_dict(),
    )

    logger.info("=" * 60)
    logger.info("QTMRL 训练开始")
    logger.info("=" * 60)

    # ========== 1. 加载数据 ==========
    logger.info("\n步骤 1: 加载预处理数据")

    metadata = load_json("data/processed/metadata.json")
    n_assets = metadata["n_assets"]
    n_features = metadata["n_features"]

    logger.info(f"资产数量: {n_assets}")
    logger.info(f"特征数量: {n_features}")
    logger.info(f"窗口大小: {config.window}")

    X_train = np.load("data/processed/X_train.npy")
    Close_train = np.load("data/processed/Close_train.npy")
    dates_train = np.load("data/processed/dates_train.npy", allow_pickle=True)

    X_valid = np.load("data/processed/X_valid.npy")
    Close_valid = np.load("data/processed/Close_valid.npy")
    dates_valid = np.load("data/processed/dates_valid.npy", allow_pickle=True)

    logger.info(f"训练集: {X_train.shape}")
    logger.info(f"验证集: {X_valid.shape}")

    # ========== 2. 创建环境 ==========
    logger.info("\n步骤 2: 创建交易环境")

    train_env = TradingEnv(
        X=X_train,
        Close=Close_train,
        dates=dates_train,
        window=config.window,
        initial_cash=config.initial_cash,
        fee_rate=config.fee_rate,
        buy_pct=config.buy_pct,
        sell_pct=config.sell_pct,
    )

    valid_env = TradingEnv(
        X=X_valid,
        Close=Close_valid,
        dates=dates_valid,
        window=config.window,
        initial_cash=config.initial_cash,
        fee_rate=config.fee_rate,
        buy_pct=config.buy_pct,
        sell_pct=config.sell_pct,
    )

    logger.info(f"训练环境: {len(dates_train)} 天")
    logger.info(f"验证环境: {len(dates_valid)} 天")

    # ========== 3. 创建模型 ==========
    logger.info("\n步骤 3: 创建模型")

    actor, critic = create_models(config, n_assets, n_features)

    logger.info(f"编码器: {config.model.encoder}")
    logger.info(f"模型维度: {config.model.d_model}")
    logger.info(f"层数: {config.model.n_layers}")

    # ========== 4. 创建训练器 ==========
    logger.info("\n步骤 4: 创建A2C训练器")

    trainer = A2CTrainer(
        actor=actor,
        critic=critic,
        lr_actor=config.train.lr_actor,
        lr_critic=config.train.lr_critic,
        gamma=config.train.gamma,
        entropy_coef=config.train.entropy_coef,
        value_coef=config.train.value_coef,
        grad_clip=config.train.grad_clip,
        device=device,
    )

    # 恢复训练
    start_step = 0
    if args.resume:
        logger.info(f"从checkpoint恢复训练: {args.resume}")
        trainer.load(args.resume)

    # ========== 5. 训练循环 ==========
    logger.info("\n步骤 5: 开始训练")
    logger.info("=" * 60)

    total_steps = config.train.total_env_steps
    rollout_steps = config.train.rollout_steps
    log_interval = config.train.log_interval_steps
    save_interval = config.train.save_interval_steps
    eval_interval = config.train.eval_interval_steps

    current_step = start_step
    episode = 0

    # 训练指标历史
    metrics_history = {
        "actor_loss": [],
        "value_loss": [],
        "entropy": [],
        "avg_reward": [],
    }

    pbar = tqdm(total=total_steps, initial=current_step, desc="训练进度")

    while current_step < total_steps:
        # 重置环境
        train_env.reset()
        episode += 1

        # 执行一次训练步骤
        stats = trainer.train_step(train_env, rollout_steps)

        current_step += stats["steps"]
        pbar.update(stats["steps"])

        # 记录指标
        for key in ["actor_loss", "value_loss", "entropy", "avg_reward"]:
            if key in stats:
                metrics_history[key].append(stats[key])

        # 日志记录
        if current_step % log_interval == 0 or current_step >= total_steps:
            log_metrics = {
                "step": current_step,
                "episode": episode,
                **stats,
            }
            logger.log_metrics(log_metrics, step=current_step)

        # 保存模型
        if current_step % save_interval == 0 or current_step >= total_steps:
            save_path = output_dir / f"checkpoint_step_{current_step}.pth"
            trainer.save(str(save_path))
            logger.info(f"模型已保存: {save_path}")

        # 验证集评估
        if current_step % eval_interval == 0 or current_step >= total_steps:
            logger.info(f"\n{'='*60}")
            logger.info(f"验证集评估 (步数: {current_step})")
            logger.info(f"{'='*60}")

            metrics, _, _ = run_backtest(
                valid_env, actor, device=device, deterministic=True
            )

            print_metrics(metrics, prefix="  ")

            # 记录到wandb
            eval_metrics = {f"eval/{k}": v for k, v in metrics.items()}
            logger.log_metrics(eval_metrics, step=current_step)

            logger.info(f"{'='*60}\n")

    pbar.close()

    # ========== 6. 保存最终模型 ==========
    logger.info("\n" + "=" * 60)
    logger.info("训练完成！")
    logger.info("=" * 60)

    final_model_path = output_dir / "final_model.pth"
    trainer.save(str(final_model_path))
    logger.info(f"最终模型已保存: {final_model_path}")

    # ========== 7. 最终评估 ==========
    logger.info("\n最终评估 - 训练集:")
    train_metrics, _, _ = run_backtest(
        train_env, actor, device=device, deterministic=True
    )
    print_metrics(train_metrics, prefix="  ")

    logger.info("\n最终评估 - 验证集:")
    valid_metrics, _, _ = run_backtest(
        valid_env, actor, device=device, deterministic=True
    )
    print_metrics(valid_metrics, prefix="  ")

    logger.finish()


if __name__ == "__main__":
    main()
