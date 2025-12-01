"""技术指标计算模块"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas_ta as ta


def calculate_sma(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """计算简单移动平均 (SMA)

    Args:
        df: 包含OHLCV数据的DataFrame
        periods: 周期列表

    Returns:
        包含SMA指标的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    for period in periods:
        result[f"SMA_{period}"] = df["Close"].rolling(window=period).mean()
    return result


def calculate_ema(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """计算指数移动平均 (EMA)

    Args:
        df: 包含OHLCV数据的DataFrame
        periods: 周期列表

    Returns:
        包含EMA指标的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    for period in periods:
        result[f"EMA_{period}"] = df["Close"].ewm(span=period, adjust=False).mean()
    return result


def calculate_rsi(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """计算相对强弱指标 (RSI)

    Args:
        df: 包含OHLCV数据的DataFrame
        periods: 周期列表

    Returns:
        包含RSI指标的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    for period in periods:
        result[f"RSI_{period}"] = ta.rsi(df["Close"], length=period)
    return result


def calculate_macd(
    df: pd.DataFrame, fast: int, slow: int, signal: int
) -> pd.DataFrame:
    """计算MACD指标

    Args:
        df: 包含OHLCV数据的DataFrame
        fast: 快速EMA周期
        slow: 慢速EMA周期
        signal: 信号线周期

    Returns:
        包含MACD指标的DataFrame
    """
    macd = ta.macd(df["Close"], fast=fast, slow=slow, signal=signal)
    if macd is not None:
        macd.columns = [f"MACD", f"MACD_signal", f"MACD_hist"]
        return macd
    else:
        # 如果计算失败，返回空DataFrame
        return pd.DataFrame(
            index=df.index, columns=["MACD", "MACD_signal", "MACD_hist"]
        )


def calculate_atr(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """计算平均真实波幅 (ATR)

    Args:
        df: 包含OHLCV数据的DataFrame
        periods: 周期列表

    Returns:
        包含ATR指标的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    for period in periods:
        result[f"ATR_{period}"] = ta.atr(
            df["High"], df["Low"], df["Close"], length=period
        )
    return result


def calculate_bbands(df: pd.DataFrame, period: int, std: float) -> pd.DataFrame:
    """计算布林带 (Bollinger Bands)

    Args:
        df: 包含OHLCV数据的DataFrame
        period: 周期
        std: 标准差倍数

    Returns:
        包含布林带指标的DataFrame
    """
    bbands = ta.bbands(df["Close"], length=period, std=std)
    if bbands is not None and len(bbands.columns) >= 3:
        # pandas_ta返回的列名格式可能是BBL_20_2, BBM_20_2, BBU_20_2 (整数)
        # 或 BBL_20_2.0, BBM_20_2.0, BBU_20_2.0 (浮点数)
        # 我们需要找到包含BBL, BBM, BBU的列
        cols = bbands.columns.tolist()
        lower_col = [c for c in cols if 'BBL' in c][0] if any('BBL' in c for c in cols) else cols[0]
        middle_col = [c for c in cols if 'BBM' in c][0] if any('BBM' in c for c in cols) else cols[1]
        upper_col = [c for c in cols if 'BBU' in c][0] if any('BBU' in c for c in cols) else cols[2]

        result = pd.DataFrame(index=df.index)
        result["BB_lower"] = bbands[lower_col]
        result["BB_middle"] = bbands[middle_col]
        result["BB_upper"] = bbands[upper_col]
        return result
    else:
        return pd.DataFrame(
            index=df.index, columns=["BB_lower", "BB_middle", "BB_upper"]
        )


def calculate_ichimoku(
    df: pd.DataFrame, tenkan: int, kijun: int, senkou: int
) -> pd.DataFrame:
    """计算一目均衡表 (Ichimoku Cloud)

    Args:
        df: 包含OHLCV数据的DataFrame
        tenkan: 转换线周期
        kijun: 基准线周期
        senkou: 先行线周期

    Returns:
        包含一目均衡表指标的DataFrame
    """
    try:
        ichimoku = ta.ichimoku(
            df["High"], df["Low"], df["Close"], tenkan=tenkan, kijun=kijun, senkou=senkou
        )
        if ichimoku is not None and len(ichimoku) > 0:
            # pandas_ta的ichimoku返回格式可能不同，使用列搜索
            ichi_df = ichimoku[0] if isinstance(ichimoku, list) else ichimoku
            cols = ichi_df.columns.tolist()

            result = pd.DataFrame(index=df.index)
            # 查找包含ITS, IKS, ISA, ISB的列
            its_col = [c for c in cols if 'ITS' in c or 'TENKAN' in c.upper()]
            iks_col = [c for c in cols if 'IKS' in c or 'KIJUN' in c.upper()]
            isa_col = [c for c in cols if 'ISA' in c or 'SENKOU_A' in c.upper() or 'SPAN_A' in c.upper()]
            isb_col = [c for c in cols if 'ISB' in c or 'SENKOU_B' in c.upper() or 'SPAN_B' in c.upper()]

            result["ICHI_tenkan"] = ichi_df[its_col[0]] if its_col else np.nan
            result["ICHI_kijun"] = ichi_df[iks_col[0]] if iks_col else np.nan
            result["ICHI_senkou_a"] = ichi_df[isa_col[0]] if isa_col else np.nan
            result["ICHI_senkou_b"] = ichi_df[isb_col[0]] if isb_col else np.nan
            return result
    except:
        pass

    return pd.DataFrame(
        index=df.index,
        columns=["ICHI_tenkan", "ICHI_kijun", "ICHI_senkou_a", "ICHI_senkou_b"],
    )


def calculate_heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """计算平均K线 (Heikin-Ashi)

    Args:
        df: 包含OHLCV数据的DataFrame

    Returns:
        包含Heikin-Ashi指标的DataFrame
    """
    ha = ta.ha(df["Open"], df["High"], df["Low"], df["Close"])
    if ha is not None:
        ha.columns = ["HA_open", "HA_high", "HA_low", "HA_close"]
        return ha
    else:
        return pd.DataFrame(
            index=df.index, columns=["HA_open", "HA_high", "HA_low", "HA_close"]
        )


def calculate_supertrend(
    df: pd.DataFrame, period: int, multiplier: float
) -> pd.DataFrame:
    """计算超级趋势 (SuperTrend)

    Args:
        df: 包含OHLCV数据的DataFrame
        period: ATR周期
        multiplier: ATR乘数

    Returns:
        包含SuperTrend指标的DataFrame
    """
    try:
        supertrend = ta.supertrend(
            df["High"], df["Low"], df["Close"], length=period, multiplier=multiplier
        )
        if supertrend is not None and len(supertrend.columns) >= 2:
            cols = supertrend.columns.tolist()
            # 查找包含SUPERT的列（主值和方向）
            supert_col = [c for c in cols if 'SUPERT_' in c and 'd' not in c]
            supertd_col = [c for c in cols if 'SUPERTd' in c or 'SUPERT_D' in c]

            result = pd.DataFrame(index=df.index)
            result["SUPERTREND"] = supertrend[supert_col[0]] if supert_col else np.nan
            result["SUPERTREND_direction"] = supertrend[supertd_col[0]] if supertd_col else np.nan
            return result
    except:
        pass

    return pd.DataFrame(
        index=df.index, columns=["SUPERTREND", "SUPERTREND_direction"]
    )


def calculate_all_indicators(
    df: pd.DataFrame, config: Dict
) -> pd.DataFrame:
    """根据配置计算所有技术指标

    Args:
        df: 包含OHLCV数据的DataFrame
        config: 指标配置字典

    Returns:
        包含所有指标的DataFrame
    """
    result = df.copy()

    # SMA
    if "sma" in config:
        sma = calculate_sma(df, config["sma"])
        result = pd.concat([result, sma], axis=1)

    # EMA
    if "ema" in config:
        ema = calculate_ema(df, config["ema"])
        result = pd.concat([result, ema], axis=1)

    # RSI
    if "rsi" in config:
        rsi = calculate_rsi(df, config["rsi"])
        result = pd.concat([result, rsi], axis=1)

    # MACD
    if "macd" in config:
        macd_params = config["macd"]
        macd = calculate_macd(df, macd_params[0], macd_params[1], macd_params[2])
        result = pd.concat([result, macd], axis=1)

    # ATR
    if "atr" in config:
        atr = calculate_atr(df, config["atr"])
        result = pd.concat([result, atr], axis=1)

    # Bollinger Bands
    if "bbands" in config:
        bb_params = config["bbands"]
        bbands = calculate_bbands(df, bb_params[0], bb_params[1])
        result = pd.concat([result, bbands], axis=1)

    # Ichimoku
    if "ichimoku" in config:
        ichi_params = config["ichimoku"]
        ichimoku = calculate_ichimoku(df, ichi_params[0], ichi_params[1], ichi_params[2])
        result = pd.concat([result, ichimoku], axis=1)

    # Heikin-Ashi
    if "heikin_ashi" in config and config["heikin_ashi"]:
        ha = calculate_heikin_ashi(df)
        result = pd.concat([result, ha], axis=1)

    # SuperTrend
    if "supertrend" in config:
        st_params = config["supertrend"]
        supertrend = calculate_supertrend(df, st_params[0], st_params[1])
        result = pd.concat([result, supertrend], axis=1)

    return result


def normalize_features(
    train_df: pd.DataFrame,
    valid_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    method: str = "zscore",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], Dict]:
    """标准化特征（仅在训练集上拟合）

    Args:
        train_df: 训练集DataFrame
        valid_df: 验证集DataFrame
        test_df: 测试集DataFrame
        feature_cols: 要标准化的特征列（None则标准化所有数值列）
        method: 标准化方法 ('zscore' 或 'minmax')

    Returns:
        (标准化后的train_df, valid_df, test_df, 标准化参数字典)
    """
    if feature_cols is None:
        # 排除非数值列
        feature_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    stats = {}

    if method == "zscore":
        # 在训练集上计算均值和标准差
        stats["mean"] = train_df[feature_cols].mean()
        stats["std"] = train_df[feature_cols].std()

        # 避免除以0
        stats["std"] = stats["std"].replace(0, 1)

        # 标准化
        train_df = train_df.copy()
        train_df[feature_cols] = (
            train_df[feature_cols] - stats["mean"]
        ) / stats["std"]

        if valid_df is not None:
            valid_df = valid_df.copy()
            valid_df[feature_cols] = (
                valid_df[feature_cols] - stats["mean"]
            ) / stats["std"]

        if test_df is not None:
            test_df = test_df.copy()
            test_df[feature_cols] = (
                test_df[feature_cols] - stats["mean"]
            ) / stats["std"]

    elif method == "minmax":
        # 在训练集上计算最小值和最大值
        stats["min"] = train_df[feature_cols].min()
        stats["max"] = train_df[feature_cols].max()

        # 避免除以0
        range_val = stats["max"] - stats["min"]
        range_val = range_val.replace(0, 1)

        # 标准化
        train_df = train_df.copy()
        train_df[feature_cols] = (train_df[feature_cols] - stats["min"]) / range_val

        if valid_df is not None:
            valid_df = valid_df.copy()
            valid_df[feature_cols] = (valid_df[feature_cols] - stats["min"]) / range_val

        if test_df is not None:
            test_df = test_df.copy()
            test_df[feature_cols] = (test_df[feature_cols] - stats["min"]) / range_val

    else:
        raise ValueError(f"不支持的标准化方法: {method}")

    return train_df, valid_df, test_df, stats
