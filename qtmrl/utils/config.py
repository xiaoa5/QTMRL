"""配置文件加载工具"""
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """配置类，支持字典式和属性式访问"""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        # 递归转换嵌套字典
        for key, value in config_dict.items():
            if isinstance(value, dict):
                self.__dict__[key] = Config(value)
            else:
                self.__dict__[key] = value

    def __getitem__(self, key):
        return self._config[key]

    def __setitem__(self, key, value):
        self._config[key] = value
        self.__dict__[key] = value

    def get(self, key, default=None):
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self._config.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self):
        return f"Config({self._config})"


def load_config(config_path: str) -> Config:
    """从YAML文件加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        Config对象
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def save_config(config: Config, save_path: str):
    """保存配置到YAML文件

    Args:
        config: Config对象
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)
