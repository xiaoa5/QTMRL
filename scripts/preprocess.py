"""数据预处理脚本

支持两种指标计算模式：
1. 基础模式：使用 indicators.py 中的指标（默认）
2. 扩展模式：使用 indicators_extended.py 中的完整指标集（当配置中包含 extended 字段时）
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import pandas as pd
from qtmrl.utils import load_config, save_json, save_dataframe
from qtmrl.dataset import StockDataset, reshape_to_tensor
from qtmrl.indicators import calculate_all_indicators, normalize_features

# 尝试导入扩展指标模块
try:
    from qtmrl.indicators_extended import calculate_all_extended_indicators
    HAS_EXTENDED_INDICATORS = True
except ImportError:
    HAS_EXTENDED_INDICATORS = False


def main():
    parser = argparse.ArgumentParser(description="数据预处理")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="强制重新下载数据",
    )
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    logging.basicConfig(level=logging.INFO)

    # 创建数据目录
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # ========== 1. 下载数据 ==========
    logging.info("=" * 50)
    logging.info("步骤 1: 下载股票数据")
    logging.info("=" * 50)

    # 获取数据的总日期范围（覆盖所有split）
    all_dates = []
    for split_name in ["train", "valid", "test"]:
        all_dates.extend(config.split[split_name])

    start_date = min(all_dates)
    end_date = max(all_dates)

    dataset = StockDataset(
        assets=config.assets,
        start_date=start_date,
        end_date=end_date,
        data_dir="data/raw",
    )

    # 下载并对齐数据
    dataset.download_data(force_download=args.force_download)
    aligned_data = dataset.align_data()

    logging.info(f"数据形状: {aligned_data.shape}")
    logging.info(f"资产数量: {len(config.assets)}")
    logging.info(f"日期范围: {dataset.get_date_range()}")

    # ========== 2. 计算技术指标 ==========
    logging.info("\n" + "=" * 50)
    logging.info("步骤 2: 计算技术指标")
    logging.info("=" * 50)

    # 检查是否使用扩展指标
    use_extended = (
        HAS_EXTENDED_INDICATORS 
        and config.features.get("extended") is not None
    )
    
    if use_extended:
        logging.info("使用扩展指标模式 (indicators_extended)")
        # 合并基础指标和扩展指标配置
        extended_config = {}
        if config.features.get("indicators"):
            extended_config.update(config.features["indicators"])
        extended_config["extended"] = config.features["extended"]
    else:
        logging.info("使用基础指标模式 (indicators)")

    # 对每个资产计算指标
    all_data = []
    for asset in config.assets:
        logging.info(f"计算 {asset} 的指标...")
        asset_data = aligned_data[aligned_data["asset"] == asset].copy()
        asset_data = asset_data.set_index("date").sort_index()

        # 计算指标
        if use_extended:
            # 使用扩展指标计算
            asset_data = calculate_all_extended_indicators(
                asset_data, extended_config
            )
        elif config.features.get("indicators"):
            # 使用基础指标计算
            asset_data = calculate_all_indicators(
                asset_data, config.features["indicators"]
            )

        # 重置索引并添加asset列
        asset_data = asset_data.reset_index()
        asset_data["asset"] = asset
        all_data.append(asset_data)

    # 合并所有资产数据
    full_data = pd.concat(all_data, ignore_index=True)

    # 处理NaN值 - 改进策略
    initial_rows = len(full_data)
    
    # 1. 获取需要处理的数值列（排除date和asset）
    numeric_cols = full_data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # 2. 按资产分组，使用前向填充 + 后向填充处理NaN
    logging.info("使用前向/后向填充处理NaN值...")
    for asset in config.assets:
        mask = full_data["asset"] == asset
        # 前向填充（用前一天的值填充）
        full_data.loc[mask, numeric_cols] = full_data.loc[mask, numeric_cols].ffill()
        # 后向填充（处理开头的NaN）
        full_data.loc[mask, numeric_cols] = full_data.loc[mask, numeric_cols].bfill()
    
    # 3. 对于仍然存在NaN的列，用0填充（极少数情况）
    remaining_nan = full_data[numeric_cols].isna().sum().sum()
    if remaining_nan > 0:
        logging.info(f"剩余 {remaining_nan} 个NaN值，用0填充")
        full_data[numeric_cols] = full_data[numeric_cols].fillna(0)
    
    # 4. 删除预热期 - 每个资产删除前N行（基于最长周期指标）
    warmup_period = 60  # 比Ichimoku的52天稍多一点
    logging.info(f"移除每个资产的前 {warmup_period} 行预热期数据...")
    
    filtered_data = []
    for asset in config.assets:
        asset_data = full_data[full_data["asset"] == asset].copy()
        if len(asset_data) > warmup_period:
            filtered_data.append(asset_data.iloc[warmup_period:])
        else:
            logging.warning(f"{asset} 数据不足，跳过预热期删除")
            filtered_data.append(asset_data)
    
    full_data = pd.concat(filtered_data, ignore_index=True)
    dropped_rows = initial_rows - len(full_data)
    logging.info(f"共移除了 {dropped_rows} 行数据（预热期）")

    # ========== 3. 分割数据集 ==========
    logging.info("\n" + "=" * 50)
    logging.info("步骤 3: 分割数据集")
    logging.info("=" * 50)

    # 处理时区信息：转换为UTC或移除时区
    full_data["date"] = pd.to_datetime(full_data["date"], utc=True)

    train_df = full_data[
        (full_data["date"] >= config.split["train"][0])
        & (full_data["date"] <= config.split["train"][1])
    ].copy()

    valid_df = full_data[
        (full_data["date"] >= config.split["valid"][0])
        & (full_data["date"] <= config.split["valid"][1])
    ].copy()

    test_df = full_data[
        (full_data["date"] >= config.split["test"][0])
        & (full_data["date"] <= config.split["test"][1])
    ].copy()

    logging.info(f"训练集: {len(train_df)} 行")
    logging.info(f"验证集: {len(valid_df)} 行")
    logging.info(f"测试集: {len(test_df)} 行")

    # ========== 4. 特征标准化 ==========
    logging.info("\n" + "=" * 50)
    logging.info("步骤 4: 特征标准化")
    logging.info("=" * 50)

    # 确定要标准化的特征列（排除date和asset）
    exclude_cols = ["date", "asset"]
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    logging.info(f"标准化 {len(feature_cols)} 个特征")
    logging.info(f"方法: {config.normalization['method']}")

    # 标准化
    train_df, valid_df, test_df, norm_stats = normalize_features(
        train_df,
        valid_df,
        test_df,
        feature_cols=feature_cols,
        method=config.normalization["method"],
    )

    # ========== 5. 转换为张量格式 ==========
    logging.info("\n" + "=" * 50)
    logging.info("步骤 5: 转换为张量格式")
    logging.info("=" * 50)

    # 特征列（用于张量）
    tensor_feature_cols = feature_cols

    X_train, Close_train, dates_train = reshape_to_tensor(
        train_df, config.assets, tensor_feature_cols
    )
    X_valid, Close_valid, dates_valid = reshape_to_tensor(
        valid_df, config.assets, tensor_feature_cols
    )
    X_test, Close_test, dates_test = reshape_to_tensor(
        test_df, config.assets, tensor_feature_cols
    )

    logging.info(f"训练集张量形状: X={X_train.shape}, Close={Close_train.shape}")
    logging.info(f"验证集张量形状: X={X_valid.shape}, Close={Close_valid.shape}")
    logging.info(f"测试集张量形状: X={X_test.shape}, Close={Close_test.shape}")

    # ========== 6. 保存数据 ==========
    logging.info("\n" + "=" * 50)
    logging.info("步骤 6: 保存处理后的数据")
    logging.info("=" * 50)

    import numpy as np

    # 保存张量数据
    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/Close_train.npy", Close_train)
    np.save("data/processed/dates_train.npy", dates_train)

    np.save("data/processed/X_valid.npy", X_valid)
    np.save("data/processed/Close_valid.npy", Close_valid)
    np.save("data/processed/dates_valid.npy", dates_valid)

    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/Close_test.npy", Close_test)
    np.save("data/processed/dates_test.npy", dates_test)

    logging.info("张量数据已保存到 data/processed/*.npy")

    # 保存元数据
    def safe_date_range(dates):
        """安全获取日期范围"""
        if len(dates) > 0:
            return [str(dates[0]), str(dates[-1])]
        return ["N/A", "N/A"]
    
    metadata = {
        "assets": config.assets,
        "n_assets": len(config.assets),
        "n_features": len(tensor_feature_cols),
        "feature_names": tensor_feature_cols,
        "window": config.window,
        "splits": {
            "train": {
                "date_range": safe_date_range(dates_train),
                "n_days": len(dates_train),
            },
            "valid": {
                "date_range": safe_date_range(dates_valid),
                "n_days": len(dates_valid),
            },
            "test": {
                "date_range": safe_date_range(dates_test),
                "n_days": len(dates_test),
            },
        },
        "normalization": {
            "method": config.normalization["method"],
            "stats": {
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in norm_stats.items()
            },
        },
    }

    save_json(metadata, "data/processed/metadata.json")
    logging.info("元数据已保存到 data/processed/metadata.json")

    logging.info("\n" + "=" * 50)
    logging.info("数据预处理完成！")
    logging.info("=" * 50)
    logging.info(f"资产数量: {metadata['n_assets']}")
    logging.info(f"特征数量: {metadata['n_features']}")
    logging.info(f"窗口大小: {metadata['window']}")
    logging.info(f"训练集: {metadata['splits']['train']['n_days']} 天")
    logging.info(f"验证集: {metadata['splits']['valid']['n_days']} 天")
    logging.info(f"测试集: {metadata['splits']['test']['n_days']} 天")


if __name__ == "__main__":
    main()
