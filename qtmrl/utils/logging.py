"""日志和实验跟踪工具"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json


class Logger:
    """统一的日志记录器，支持文件日志和Wandb"""

    def __init__(
        self,
        output_dir: str,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_name: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
    ):
        """初始化日志记录器

        Args:
            output_dir: 输出目录
            use_wandb: 是否使用Wandb
            wandb_project: Wandb项目名
            wandb_entity: Wandb实体名
            wandb_name: Wandb运行名称
            wandb_config: Wandb配置
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置Python logging
        self._setup_file_logging()

        # 设置Wandb
        self.use_wandb = use_wandb
        self.wandb = None
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=wandb_name,
                    config=wandb_config,
                    dir=str(self.output_dir),
                )
                logging.info("Wandb初始化成功")
            except ImportError:
                logging.warning("未安装wandb，跳过wandb日志记录")
                self.use_wandb = False
            except Exception as e:
                logging.warning(f"Wandb初始化失败: {e}")
                self.use_wandb = False

    def _setup_file_logging(self):
        """设置文件日志"""
        log_file = self.output_dir / "train.log"

        # 配置logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """记录指标

        Args:
            metrics: 指标字典
            step: 步数
        """
        # 记录到文件
        if step is not None:
            log_str = f"Step {step}: " + ", ".join(
                [f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
            )
        else:
            log_str = ", ".join(
                [f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
            )
        logging.info(log_str)

        # 记录到Wandb
        if self.use_wandb and self.wandb is not None:
            self.wandb.log(metrics, step=step)

    def log_config(self, config: Dict[str, Any]):
        """记录配置

        Args:
            config: 配置字典
        """
        config_file = self.output_dir / "config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logging.info(f"配置已保存到 {config_file}")

        if self.use_wandb and self.wandb is not None:
            self.wandb.config.update(config)

    def info(self, message: str):
        """记录信息"""
        logging.info(message)

    def warning(self, message: str):
        """记录警告"""
        logging.warning(message)

    def error(self, message: str):
        """记录错误"""
        logging.error(message)

    def save_model(self, model_path: str):
        """保存模型到Wandb

        Args:
            model_path: 模型文件路径
        """
        if self.use_wandb and self.wandb is not None:
            self.wandb.save(model_path)

    def finish(self):
        """结束日志记录"""
        if self.use_wandb and self.wandb is not None:
            self.wandb.finish()
