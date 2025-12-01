"""回测模块"""
import torch
import numpy as np
from typing import Dict, Tuple
from qtmrl.env import TradingEnv
from .metrics import calculate_all_metrics


def run_backtest(
    env: TradingEnv,
    actor: torch.nn.Module,
    device: str = "cpu",
    deterministic: bool = True,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """运行回测

    Args:
        env: 交易环境
        actor: Actor模型
        device: 设备
        deterministic: 是否使用确定性策略

    Returns:
        (metrics, portfolio_values, actions_history)
        - metrics: 评估指标字典
        - portfolio_values: 组合价值序列
        - actions_history: 动作历史
    """
    actor.eval()
    actor.to(device)

    # 重置环境
    state = env.reset()

    done = False
    actions_history = []

    with torch.no_grad():
        while not done:
            # 转换状态为tensor
            features = torch.from_numpy(state["features"]).unsqueeze(0).to(device)
            positions = torch.from_numpy(state["positions"]).unsqueeze(0).to(device)
            cash = torch.from_numpy(state["cash"]).unsqueeze(0).to(device)

            # 采样动作
            actions, _ = actor.sample_action(
                features, positions, cash, deterministic=deterministic
            )

            # 执行动作
            actions_np = actions.cpu().numpy()[0]
            state, reward, done, info = env.step(actions_np)

            actions_history.append(actions_np)

    # 获取组合价值历史
    portfolio_values = env.get_portfolio_values()

    # 计算指标
    metrics = calculate_all_metrics(portfolio_values, annualize=True)

    return metrics, portfolio_values, np.array(actions_history)


def run_multiple_backtests(
    env: TradingEnv,
    actor: torch.nn.Module,
    n_runs: int = 5,
    device: str = "cpu",
) -> Dict:
    """运行多次回测并计算统计量

    Args:
        env: 交易环境
        actor: Actor模型
        n_runs: 运行次数
        device: 设备

    Returns:
        统计量字典
    """
    all_metrics = []

    for i in range(n_runs):
        metrics, _, _ = run_backtest(
            env, actor, device=device, deterministic=False
        )
        all_metrics.append(metrics)

    # 计算统计量
    stats = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        stats[f"{key}_mean"] = np.mean(values)
        stats[f"{key}_std"] = np.std(values)
        stats[f"{key}_min"] = np.min(values)
        stats[f"{key}_max"] = np.max(values)

    return stats
