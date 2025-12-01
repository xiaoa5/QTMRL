"""评估指标计算"""
import numpy as np
from typing import Dict


def calculate_total_return(portfolio_values: np.ndarray) -> float:
    """计算总收益率

    Args:
        portfolio_values: 组合价值序列

    Returns:
        总收益率
    """
    if len(portfolio_values) < 2:
        return 0.0

    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]

    if initial_value == 0:
        return 0.0

    return (final_value / initial_value) - 1.0


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """计算夏普比率

    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率（日频）

    Returns:
        夏普比率
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate
    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns)

    if std_return == 0:
        return 0.0

    return mean_return / std_return


def calculate_volatility(returns: np.ndarray) -> float:
    """计算波动率

    Args:
        returns: 收益率序列

    Returns:
        波动率（标准差）
    """
    if len(returns) == 0:
        return 0.0

    return np.std(returns)


def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
    """计算最大回撤

    Args:
        portfolio_values: 组合价值序列

    Returns:
        最大回撤（负值）
    """
    if len(portfolio_values) < 2:
        return 0.0

    # 计算累计最大值
    cummax = np.maximum.accumulate(portfolio_values)

    # 计算回撤
    drawdowns = (portfolio_values - cummax) / cummax

    # 最大回撤
    max_dd = np.min(drawdowns)

    return max_dd


def calculate_annualized_return(
    total_return: float, n_days: int, trading_days_per_year: int = 252
) -> float:
    """计算年化收益率

    Args:
        total_return: 总收益率
        n_days: 交易天数
        trading_days_per_year: 每年交易日数

    Returns:
        年化收益率
    """
    if n_days == 0:
        return 0.0

    years = n_days / trading_days_per_year
    annualized = (1 + total_return) ** (1 / years) - 1

    return annualized


def calculate_annualized_volatility(
    volatility: float, trading_days_per_year: int = 252
) -> float:
    """计算年化波动率

    Args:
        volatility: 日频波动率
        trading_days_per_year: 每年交易日数

    Returns:
        年化波动率
    """
    return volatility * np.sqrt(trading_days_per_year)


def calculate_all_metrics(
    portfolio_values: np.ndarray, annualize: bool = False
) -> Dict[str, float]:
    """计算所有评估指标

    Args:
        portfolio_values: 组合价值序列
        annualize: 是否年化指标

    Returns:
        指标字典
    """
    # 计算收益率序列
    if len(portfolio_values) < 2:
        returns = np.array([])
    else:
        returns = np.diff(portfolio_values) / portfolio_values[:-1]

    # 计算指标
    total_return = calculate_total_return(portfolio_values)
    sharpe = calculate_sharpe_ratio(returns)
    volatility = calculate_volatility(returns)
    max_dd = calculate_max_drawdown(portfolio_values)

    metrics = {
        "total_return": total_return,
        "sharpe": sharpe,
        "volatility": volatility,
        "max_drawdown": max_dd,
        "final_value": portfolio_values[-1] if len(portfolio_values) > 0 else 0.0,
        "n_days": len(portfolio_values) - 1 if len(portfolio_values) > 1 else 0,
    }

    # 年化指标
    if annualize and metrics["n_days"] > 0:
        metrics["annualized_return"] = calculate_annualized_return(
            total_return, metrics["n_days"]
        )
        metrics["annualized_volatility"] = calculate_annualized_volatility(volatility)
        metrics["annualized_sharpe"] = (
            metrics["annualized_return"] / metrics["annualized_volatility"]
            if metrics["annualized_volatility"] > 0
            else 0.0
        )

    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """打印指标

    Args:
        metrics: 指标字典
        prefix: 前缀字符串
    """
    print(f"{prefix}评估指标:")
    print(f"{prefix}  总收益率: {metrics['total_return']*100:.2f}%")
    print(f"{prefix}  夏普比率: {metrics['sharpe']:.4f}")
    print(f"{prefix}  波动率: {metrics['volatility']*100:.2f}%")
    print(f"{prefix}  最大回撤: {metrics['max_drawdown']*100:.2f}%")
    print(f"{prefix}  最终价值: ${metrics['final_value']:.2f}")
    print(f"{prefix}  交易天数: {metrics['n_days']}")

    if "annualized_return" in metrics:
        print(f"\n{prefix}年化指标:")
        print(f"{prefix}  年化收益率: {metrics['annualized_return']*100:.2f}%")
        print(f"{prefix}  年化波动率: {metrics['annualized_volatility']*100:.2f}%")
        print(f"{prefix}  年化夏普比率: {metrics['annualized_sharpe']:.4f}")
