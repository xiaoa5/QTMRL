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


def calculate_sharpe_ratio(
    returns: np.ndarray, risk_free_rate: float = 0.0, annualize: bool = False,
    trading_days_per_year: int = 252
) -> float:
    """计算夏普比率

    Args:
        returns: 日收益率序列
        risk_free_rate: 无风险利率（日频）
        annualize: 是否年化（乘以sqrt(252)）
        trading_days_per_year: 每年交易日数

    Returns:
        夏普比率（日频或年化）
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate
    mean_return = np.mean(excess_returns)
    # 使用样本标准差 (ddof=1)，更适合有限样本
    std_return = np.std(excess_returns, ddof=1) if len(returns) > 1 else 0.0

    if std_return == 0:
        return 0.0

    daily_sharpe = mean_return / std_return

    if annualize:
        return daily_sharpe * np.sqrt(trading_days_per_year)

    return daily_sharpe


def calculate_volatility(
    returns: np.ndarray, annualize: bool = False, trading_days_per_year: int = 252
) -> float:
    """计算波动率

    Args:
        returns: 日收益率序列
        annualize: 是否年化
        trading_days_per_year: 每年交易日数

    Returns:
        波动率（标准差）
    """
    if len(returns) == 0:
        return 0.0

    # 使用样本标准差 (ddof=1)
    vol = np.std(returns, ddof=1) if len(returns) > 1 else 0.0

    if annualize:
        return vol * np.sqrt(trading_days_per_year)

    return vol


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

    # 避免除以零
    safe_cummax = np.where(cummax == 0, 1.0, cummax)

    # 计算回撤
    drawdowns = (portfolio_values - cummax) / safe_cummax

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

    # 防止负基数导致复数
    if total_return <= -1.0:
        return -1.0

    annualized = (1 + total_return) ** (1 / years) - 1

    return annualized


def calculate_all_metrics(
    portfolio_values: np.ndarray, annualize: bool = False
) -> Dict[str, float]:
    """计算所有评估指标

    Args:
        portfolio_values: 组合价值序列
        annualize: 是否计算年化指标

    Returns:
        指标字典
    """
    # 计算收益率序列
    if len(portfolio_values) < 2:
        returns = np.array([])
    else:
        # 防止除以零
        safe_values = np.where(portfolio_values[:-1] == 0, 1.0, portfolio_values[:-1])
        returns = np.diff(portfolio_values) / safe_values

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
        metrics["annualized_volatility"] = calculate_volatility(
            returns, annualize=True
        )
        # 正确的年化Sharpe: daily_sharpe * sqrt(252)
        metrics["annualized_sharpe"] = calculate_sharpe_ratio(
            returns, annualize=True
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
