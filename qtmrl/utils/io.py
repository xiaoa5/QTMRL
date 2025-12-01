"""文件读写工具"""
import json
import pickle
from pathlib import Path
from typing import Any, Dict
import pandas as pd
import torch


def save_json(data: Dict, path: str):
    """保存JSON文件

    Args:
        data: 数据字典
        path: 保存路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> Dict:
    """加载JSON文件

    Args:
        path: 文件路径

    Returns:
        数据字典
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(data: Any, path: str):
    """保存pickle文件

    Args:
        data: 数据
        path: 保存路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path: str) -> Any:
    """加载pickle文件

    Args:
        path: 文件路径

    Returns:
        数据
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: str,
    **kwargs,
):
    """保存训练checkpoint

    Args:
        model: 模型
        optimizer: 优化器
        step: 当前步数
        path: 保存路径
        **kwargs: 其他要保存的信息
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        **kwargs,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str) -> Dict:
    """加载训练checkpoint

    Args:
        path: checkpoint路径

    Returns:
        checkpoint字典
    """
    return torch.load(path, map_location="cpu")


def save_dataframe(df: pd.DataFrame, path: str, format: str = "parquet"):
    """保存DataFrame

    Args:
        df: DataFrame
        path: 保存路径
        format: 格式 ('parquet', 'csv', 'hdf5')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        df.to_parquet(path)
    elif format == "csv":
        df.to_csv(path, index=True)
    elif format == "hdf5":
        df.to_hdf(path, key="data", mode="w")
    else:
        raise ValueError(f"不支持的格式: {format}")


def load_dataframe(path: str, format: str = "parquet") -> pd.DataFrame:
    """加载DataFrame

    Args:
        path: 文件路径
        format: 格式 ('parquet', 'csv', 'hdf5')

    Returns:
        DataFrame
    """
    if format == "parquet":
        return pd.read_parquet(path)
    elif format == "csv":
        return pd.read_csv(path, index_col=0, parse_dates=True)
    elif format == "hdf5":
        return pd.read_hdf(path, key="data")
    else:
        raise ValueError(f"不支持的格式: {format}")
