"""数据集管理模块"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import yfinance as yf
from datetime import datetime
import logging


class StockDataset:
    """股票数据集类"""

    def __init__(
        self,
        assets: List[str],
        start_date: str,
        end_date: str,
        data_dir: str = "data/raw",
    ):
        """初始化数据集

        Args:
            assets: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            data_dir: 数据目录
        """
        self.assets = assets
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.data: Optional[Dict[str, pd.DataFrame]] = None
        self.aligned_data: Optional[pd.DataFrame] = None

    def download_data(self, force_download: bool = False) -> Dict[str, pd.DataFrame]:
        """从Yahoo Finance下载数据

        Args:
            force_download: 是否强制重新下载

        Returns:
            字典，键为股票代码，值为DataFrame
        """
        self.data = {}

        for asset in self.assets:
            csv_file = self.data_dir / f"{asset}.csv"

            # 如果文件存在且不强制下载，则加载本地文件
            if csv_file.exists() and not force_download:
                logging.info(f"从本地加载 {asset} 数据")
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            else:
                logging.info(f"从Yahoo Finance下载 {asset} 数据")
                try:
                    ticker = yf.Ticker(asset)
                    df = ticker.history(start=self.start_date, end=self.end_date)

                    if df.empty:
                        logging.warning(f"{asset} 数据为空，跳过")
                        continue

                    # 保存到本地
                    df.to_csv(csv_file)
                    logging.info(f"保存 {asset} 数据到 {csv_file}")

                except Exception as e:
                    logging.error(f"下载 {asset} 数据失败: {e}")
                    continue

            # 确保列名标准化
            df.columns = [col.strip().capitalize() for col in df.columns]

            # 选择OHLCV列
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            if all(col in df.columns for col in required_cols):
                self.data[asset] = df[required_cols]
            else:
                logging.warning(f"{asset} 缺少必需列，跳过")

        logging.info(f"成功加载 {len(self.data)} 只股票数据")
        return self.data

    def align_data(self) -> pd.DataFrame:
        """对齐多只股票的数据（处理缺失值）

        Returns:
            多索引DataFrame，层级为(date, asset)
        """
        if self.data is None or len(self.data) == 0:
            raise ValueError("请先下载数据")

        # 合并所有股票数据
        dfs = []
        for asset, df in self.data.items():
            df = df.copy()
            df["asset"] = asset
            df.index.name = "date"
            df = df.reset_index()
            dfs.append(df)

        # 合并
        aligned = pd.concat(dfs, ignore_index=True)

        # 获取所有日期的并集
        all_dates = sorted(aligned["date"].unique())

        # 对每个资产进行前向填充
        aligned_list = []
        for asset in self.assets:
            asset_data = aligned[aligned["asset"] == asset].copy()
            asset_data = asset_data.set_index("date")

            # 重索引到所有日期
            asset_data = asset_data.reindex(all_dates)
            asset_data["asset"] = asset

            # 价格前向填充
            price_cols = ["Open", "High", "Low", "Close"]
            asset_data[price_cols] = asset_data[price_cols].ffill()

            # Volume缺失填0
            asset_data["Volume"] = asset_data["Volume"].fillna(0)

            aligned_list.append(asset_data)

        # 合并
        self.aligned_data = pd.concat(aligned_list)
        self.aligned_data = self.aligned_data.reset_index()
        self.aligned_data = self.aligned_data.rename(columns={"index": "date"})

        # 移除仍有缺失值的行（起始部分）
        initial_len = len(self.aligned_data)
        self.aligned_data = self.aligned_data.dropna()
        dropped = initial_len - len(self.aligned_data)
        if dropped > 0:
            logging.info(f"移除了 {dropped} 行包含缺失值的数据")

        logging.info(
            f"数据对齐完成: {len(self.aligned_data)} 行, "
            f"{len(self.aligned_data['date'].unique())} 个交易日, "
            f"{len(self.assets)} 只股票"
        )

        return self.aligned_data

    def get_date_range(self) -> Tuple[str, str]:
        """获取实际数据的日期范围

        Returns:
            (开始日期, 结束日期)
        """
        if self.aligned_data is None:
            raise ValueError("请先对齐数据")

        dates = self.aligned_data["date"].unique()
        return str(dates.min()), str(dates.max())

    def split_by_date(
        self, train_dates: Tuple[str, str], valid_dates: Tuple[str, str], test_dates: Tuple[str, str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """按日期分割数据集

        Args:
            train_dates: 训练集日期范围
            valid_dates: 验证集日期范围
            test_dates: 测试集日期范围

        Returns:
            (train_df, valid_df, test_df)
        """
        if self.aligned_data is None:
            raise ValueError("请先对齐数据")

        df = self.aligned_data.copy()
        df["date"] = pd.to_datetime(df["date"])

        # 分割
        train_df = df[
            (df["date"] >= train_dates[0]) & (df["date"] <= train_dates[1])
        ]
        valid_df = df[
            (df["date"] >= valid_dates[0]) & (df["date"] <= valid_dates[1])
        ]
        test_df = df[
            (df["date"] >= test_dates[0]) & (df["date"] <= test_dates[1])
        ]

        logging.info(
            f"数据分割: 训练集 {len(train_df)} 行, "
            f"验证集 {len(valid_df)} 行, "
            f"测试集 {len(test_df)} 行"
        )

        return train_df, valid_df, test_df


def reshape_to_tensor(
    df: pd.DataFrame, assets: List[str], feature_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将DataFrame重塑为张量格式

    Args:
        df: DataFrame，包含date, asset, features
        assets: 资产列表
        feature_cols: 特征列

    Returns:
        (X, Close, dates)
        - X: [T, N, F] 特征张量
        - Close: [T, N] 收盘价张量
        - dates: [T] 日期数组
    """
    # 获取所有日期
    dates = sorted(df["date"].unique())
    T = len(dates)
    N = len(assets)
    F = len(feature_cols)

    # 初始化张量
    X = np.zeros((T, N, F), dtype=np.float32)
    Close = np.zeros((T, N), dtype=np.float32)

    # 填充数据
    for t, date in enumerate(dates):
        date_data = df[df["date"] == date]
        for n, asset in enumerate(assets):
            asset_data = date_data[date_data["asset"] == asset]
            if len(asset_data) > 0:
                X[t, n, :] = asset_data[feature_cols].values[0]
                Close[t, n] = asset_data["Close"].values[0]

    return X, Close, np.array(dates)
