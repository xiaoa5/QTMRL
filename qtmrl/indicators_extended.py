"""扩展技术指标计算模块 - 基于原论文QTMRL的完整指标集

本模块扩展了原有的indicators.py，添加了原作者使用的所有技术指标：
- 基础OHLCV衍生特征 (return, volatility, price_change, volume_change)
- 额外的ATR周期 (ATR_10)
- True Range (TR)
- 标准差 (STDDEV)
- 衍生比率 (sma_ratio, bb_width)
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas_ta as ta


# =============================================================================
# 基础衍生特征 (原作者使用)
# =============================================================================

def calculate_return(df: pd.DataFrame) -> pd.DataFrame:
    """计算收益率 (Return)
    
    Args:
        df: 包含OHLCV数据的DataFrame
        
    Returns:
        包含收益率的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    result["Return"] = df["Close"].pct_change().fillna(0)
    return result


def calculate_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """计算日内波动率 (Volatility = High - Low)
    
    Args:
        df: 包含OHLCV数据的DataFrame
        
    Returns:
        包含波动率的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    result["Volatility"] = df["High"] - df["Low"]
    return result


def calculate_price_change(df: pd.DataFrame) -> pd.DataFrame:
    """计算价格变化率 (Price Change = (Close - Open) / Open)
    
    Args:
        df: 包含OHLCV数据的DataFrame
        
    Returns:
        包含价格变化率的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    result["Price_Change"] = (df["Close"] - df["Open"]) / df["Open"].replace(0, 1e-8)
    return result


def calculate_volume_change(df: pd.DataFrame) -> pd.DataFrame:
    """计算成交量变化率 (Volume Change)
    
    Args:
        df: 包含OHLCV数据的DataFrame
        
    Returns:
        包含成交量变化率的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    result["Volume_Change"] = df["Volume"].pct_change().fillna(0)
    return result


# =============================================================================
# 额外技术指标
# =============================================================================

def calculate_true_range(df: pd.DataFrame) -> pd.DataFrame:
    """计算真实波幅 (True Range)
    
    TR = max(High - Low, |High - Close_prev|, |Low - Close_prev|)
    
    Args:
        df: 包含OHLCV数据的DataFrame
        
    Returns:
        包含TR的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    
    # 计算三个候选值
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"] - df["Close"].shift(1)).abs()
    
    # True Range是三者的最大值
    result["TR"] = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return result


def calculate_stddev(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """计算收盘价标准差 (STDDEV)
    
    Args:
        df: 包含OHLCV数据的DataFrame
        periods: 周期列表
        
    Returns:
        包含STDDEV指标的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    for period in periods:
        result[f"STDDEV_{period}"] = df["Close"].rolling(window=period).std()
    return result


def calculate_sma_ratio(df: pd.DataFrame, fast: int = 5, slow: int = 20) -> pd.DataFrame:
    """计算SMA比率 (SMA Ratio = SMA_fast / SMA_slow)
    
    用于判断短期趋势相对于长期趋势的位置
    
    Args:
        df: 包含OHLCV数据的DataFrame
        fast: 快速SMA周期
        slow: 慢速SMA周期
        
    Returns:
        包含SMA比率的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    sma_fast = df["Close"].rolling(window=fast).mean()
    sma_slow = df["Close"].rolling(window=slow).mean()
    result["SMA_Ratio"] = sma_fast / sma_slow.replace(0, 1e-8)
    return result


def calculate_bb_width(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    """计算布林带宽度 (BB Width = (Upper - Lower) / Close)
    
    布林带宽度可反映市场波动性
    
    Args:
        df: 包含OHLCV数据的DataFrame
        period: 布林带周期
        std: 标准差倍数
        
    Returns:
        包含布林带宽度的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    
    bbands = ta.bbands(df["Close"], length=period, std=std)
    if bbands is not None and len(bbands.columns) >= 3:
        cols = bbands.columns.tolist()
        upper_col = [c for c in cols if 'BBU' in c]
        lower_col = [c for c in cols if 'BBL' in c]
        
        if upper_col and lower_col:
            upper = bbands[upper_col[0]]
            lower = bbands[lower_col[0]]
            result["BB_Width"] = (upper - lower) / df["Close"].replace(0, 1e-8)
        else:
            result["BB_Width"] = np.nan
    else:
        result["BB_Width"] = np.nan
    
    return result


def calculate_rsi_divergence(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算RSI背离信号 (RSI Divergence)
    
    当价格创新高但RSI未创新高时为看跌背离
    当价格创新低但RSI未创新低时为看涨背离
    
    Args:
        df: 包含OHLCV数据的DataFrame
        period: RSI周期
        
    Returns:
        包含RSI背离信号的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    
    rsi = ta.rsi(df["Close"], length=period)
    
    # 计算滚动窗口内的高低点
    price_high = df["Close"].rolling(window=period).max()
    price_low = df["Close"].rolling(window=period).min()
    rsi_high = rsi.rolling(window=period).max() if rsi is not None else pd.Series(index=df.index)
    rsi_low = rsi.rolling(window=period).min() if rsi is not None else pd.Series(index=df.index)
    
    # 背离信号: 1=看涨背离, -1=看跌背离, 0=无背离
    bearish_div = (df["Close"] == price_high) & (rsi < rsi_high)
    bullish_div = (df["Close"] == price_low) & (rsi > rsi_low)
    
    result["RSI_Divergence"] = 0
    result.loc[bearish_div, "RSI_Divergence"] = -1
    result.loc[bullish_div, "RSI_Divergence"] = 1
    
    return result


def calculate_momentum(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """计算动量指标 (Momentum)
    
    Momentum = Close - Close_n (n天前的收盘价)
    
    Args:
        df: 包含OHLCV数据的DataFrame
        periods: 周期列表
        
    Returns:
        包含动量指标的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    for period in periods:
        result[f"MOM_{period}"] = df["Close"] - df["Close"].shift(period)
    return result


def calculate_roc(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """计算变化率 (Rate of Change)
    
    ROC = (Close - Close_n) / Close_n * 100
    
    Args:
        df: 包含OHLCV数据的DataFrame
        periods: 周期列表
        
    Returns:
        包含ROC指标的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    for period in periods:
        prev_close = df["Close"].shift(period)
        result[f"ROC_{period}"] = (df["Close"] - prev_close) / prev_close.replace(0, 1e-8) * 100
    return result


def calculate_williams_r(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """计算威廉指标 (Williams %R)
    
    Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    
    Args:
        df: 包含OHLCV数据的DataFrame
        periods: 周期列表
        
    Returns:
        包含Williams %R指标的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    for period in periods:
        willr = ta.willr(df["High"], df["Low"], df["Close"], length=period)
        if willr is not None:
            result[f"WILLR_{period}"] = willr
        else:
            result[f"WILLR_{period}"] = np.nan
    return result


def calculate_cci(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
    """计算商品通道指数 (CCI)
    
    Args:
        df: 包含OHLCV数据的DataFrame
        periods: 周期列表
        
    Returns:
        包含CCI指标的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    for period in periods:
        cci = ta.cci(df["High"], df["Low"], df["Close"], length=period)
        if cci is not None:
            result[f"CCI_{period}"] = cci
        else:
            result[f"CCI_{period}"] = np.nan
    return result


def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """计算随机指标 (Stochastic Oscillator)
    
    Args:
        df: 包含OHLCV数据的DataFrame
        k_period: %K周期
        d_period: %D周期
        
    Returns:
        包含随机指标的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    
    stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=k_period, d=d_period)
    if stoch is not None and len(stoch.columns) >= 2:
        cols = stoch.columns.tolist()
        k_col = [c for c in cols if 'STOCHk' in c or 'K' in c.upper()]
        d_col = [c for c in cols if 'STOCHd' in c or 'D' in c.upper()]
        
        result["STOCH_K"] = stoch[k_col[0]] if k_col else np.nan
        result["STOCH_D"] = stoch[d_col[0]] if d_col else np.nan
    else:
        result["STOCH_K"] = np.nan
        result["STOCH_D"] = np.nan
    
    return result


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """计算平均趋向指数 (ADX)
    
    Args:
        df: 包含OHLCV数据的DataFrame
        period: ADX周期
        
    Returns:
        包含ADX指标的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    
    adx = ta.adx(df["High"], df["Low"], df["Close"], length=period)
    if adx is not None and len(adx.columns) >= 1:
        cols = adx.columns.tolist()
        adx_col = [c for c in cols if 'ADX_' in c and 'DM' not in c]
        dmp_col = [c for c in cols if 'DMP_' in c]
        dmn_col = [c for c in cols if 'DMN_' in c]
        
        result["ADX"] = adx[adx_col[0]] if adx_col else np.nan
        result["DI_Plus"] = adx[dmp_col[0]] if dmp_col else np.nan
        result["DI_Minus"] = adx[dmn_col[0]] if dmn_col else np.nan
    else:
        result["ADX"] = np.nan
        result["DI_Plus"] = np.nan
        result["DI_Minus"] = np.nan
    
    return result


def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """计算能量潮 (On-Balance Volume)
    
    Args:
        df: 包含OHLCV数据的DataFrame
        
    Returns:
        包含OBV指标的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    
    obv = ta.obv(df["Close"], df["Volume"])
    if obv is not None:
        result["OBV"] = obv
    else:
        result["OBV"] = np.nan
    
    return result


def calculate_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """计算成交量加权平均价 (VWAP)
    
    Args:
        df: 包含OHLCV数据的DataFrame
        
    Returns:
        包含VWAP指标的DataFrame
    """
    result = pd.DataFrame(index=df.index)
    
    # 典型价格
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    
    # 累积成交量加权价格
    cum_vol_price = (typical_price * df["Volume"]).cumsum()
    cum_vol = df["Volume"].cumsum()
    
    result["VWAP"] = cum_vol_price / cum_vol.replace(0, 1e-8)
    return result


# =============================================================================
# 完整的扩展指标计算函数
# =============================================================================

def calculate_all_extended_indicators(
    df: pd.DataFrame, config: Dict
) -> pd.DataFrame:
    """根据配置计算所有扩展技术指标
    
    此函数是原 indicators.py 中 calculate_all_indicators 的扩展版本，
    添加了原作者QTMRL论文中使用的所有指标。
    
    Args:
        df: 包含OHLCV数据的DataFrame
        config: 指标配置字典，可包含以下键:
            - sma: List[int] - SMA周期列表
            - ema: List[int] - EMA周期列表  
            - rsi: List[int] - RSI周期列表
            - macd: [fast, slow, signal] - MACD参数
            - atr: List[int] - ATR周期列表
            - bbands: [period, std] - 布林带参数
            - ichimoku: [tenkan, kijun, senkou] - 一目均衡表参数
            - heikin_ashi: bool - 是否计算平均K线
            - supertrend: [period, multiplier] - SuperTrend参数
            - extended: dict - 扩展指标配置:
                - return: bool - 收益率
                - volatility: bool - 波动率
                - price_change: bool - 价格变化率
                - volume_change: bool - 成交量变化率
                - tr: bool - True Range
                - stddev: List[int] - 标准差周期
                - sma_ratio: [fast, slow] - SMA比率
                - bb_width: [period, std] - 布林带宽度
                - momentum: List[int] - 动量周期
                - roc: List[int] - ROC周期
                - williams_r: List[int] - Williams %R周期
                - cci: List[int] - CCI周期
                - stochastic: [k, d] - 随机指标参数
                - adx: int - ADX周期
                - obv: bool - OBV
                - vwap: bool - VWAP
                
    Returns:
        包含所有指标的DataFrame
    """
    # 导入原有的指标计算函数
    from qtmrl.indicators import (
        calculate_sma, calculate_ema, calculate_rsi, calculate_macd,
        calculate_atr, calculate_bbands, calculate_ichimoku,
        calculate_heikin_ashi, calculate_supertrend
    )
    
    result = df.copy()
    
    # ===================
    # 原有指标
    # ===================
    
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
    
    # ATR - 支持多周期
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
    
    # ===================
    # 扩展指标 (新增)
    # ===================
    
    extended = config.get("extended", {})
    
    # 收益率
    if extended.get("return", False):
        ret = calculate_return(df)
        result = pd.concat([result, ret], axis=1)
    
    # 波动率
    if extended.get("volatility", False):
        vol = calculate_volatility(df)
        result = pd.concat([result, vol], axis=1)
    
    # 价格变化率
    if extended.get("price_change", False):
        pc = calculate_price_change(df)
        result = pd.concat([result, pc], axis=1)
    
    # 成交量变化率
    if extended.get("volume_change", False):
        vc = calculate_volume_change(df)
        result = pd.concat([result, vc], axis=1)
    
    # True Range
    if extended.get("tr", False):
        tr = calculate_true_range(df)
        result = pd.concat([result, tr], axis=1)
    
    # 标准差
    if "stddev" in extended:
        stddev = calculate_stddev(df, extended["stddev"])
        result = pd.concat([result, stddev], axis=1)
    
    # SMA比率
    if "sma_ratio" in extended:
        ratio_params = extended["sma_ratio"]
        sma_ratio = calculate_sma_ratio(df, ratio_params[0], ratio_params[1])
        result = pd.concat([result, sma_ratio], axis=1)
    
    # 布林带宽度
    if "bb_width" in extended:
        bw_params = extended["bb_width"]
        bb_width = calculate_bb_width(df, bw_params[0], bw_params[1])
        result = pd.concat([result, bb_width], axis=1)
    
    # 动量
    if "momentum" in extended:
        mom = calculate_momentum(df, extended["momentum"])
        result = pd.concat([result, mom], axis=1)
    
    # ROC
    if "roc" in extended:
        roc = calculate_roc(df, extended["roc"])
        result = pd.concat([result, roc], axis=1)
    
    # Williams %R
    if "williams_r" in extended:
        willr = calculate_williams_r(df, extended["williams_r"])
        result = pd.concat([result, willr], axis=1)
    
    # CCI
    if "cci" in extended:
        cci = calculate_cci(df, extended["cci"])
        result = pd.concat([result, cci], axis=1)
    
    # Stochastic
    if "stochastic" in extended:
        stoch_params = extended["stochastic"]
        stoch = calculate_stochastic(df, stoch_params[0], stoch_params[1])
        result = pd.concat([result, stoch], axis=1)
    
    # ADX
    if "adx" in extended:
        adx = calculate_adx(df, extended["adx"])
        result = pd.concat([result, adx], axis=1)
    
    # OBV
    if extended.get("obv", False):
        obv = calculate_obv(df)
        result = pd.concat([result, obv], axis=1)
    
    # VWAP
    if extended.get("vwap", False):
        vwap = calculate_vwap(df)
        result = pd.concat([result, vwap], axis=1)
    
    return result


# =============================================================================
# 原论文完整配置
# =============================================================================

def get_qtmrl_paper_config() -> Dict:
    """获取原论文QTMRL使用的完整指标配置
    
    这是根据原作者代码中的 self.feature_columns 定义的完整配置。
    
    Returns:
        原论文使用的完整指标配置字典
    """
    return {
        # 基础移动平均
        "sma": [5, 20, 50],
        "ema": [12, 26, 50],
        
        # 动量指标
        "rsi": [14],
        "macd": [12, 26, 9],
        
        # 波动性指标
        "atr": [10, 14],  # 原作者使用了ATR_10和ATR_14
        "bbands": [20, 2],
        
        # 趋势指标
        "ichimoku": [9, 26, 52],
        "supertrend": [10, 3],
        
        # K线形态
        "heikin_ashi": True,
        
        # 扩展指标
        "extended": {
            # 基础衍生特征
            "return": True,
            "volatility": True,
            "price_change": True,
            "volume_change": True,
            
            # 波动性
            "tr": True,
            "stddev": [20],
            
            # 衍生比率
            "sma_ratio": [5, 20],
            "bb_width": [20, 2],
        }
    }


def get_full_extended_config() -> Dict:
    """获取完整的扩展指标配置
    
    包含原论文指标 + 额外常用指标
    
    Returns:
        完整的扩展指标配置字典
    """
    config = get_qtmrl_paper_config()
    
    # 添加额外的常用指标
    config["extended"].update({
        "momentum": [10, 20],
        "roc": [10, 20],
        "williams_r": [14],
        "cci": [20],
        "stochastic": [14, 3],
        "adx": 14,
        "obv": True,
        "vwap": True,
    })
    
    return config
