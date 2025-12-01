# Changelog

All notable changes to QTMRL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for v0.2.0
- 基线策略对比 (Buy & Hold, 等权重, 动量策略)
- 早停和模型选择机制
- 风险敏感奖励函数 (Sharpe-aware, Drawdown penalty)
- 消融实验自动化脚本
- 增强可视化 (持仓热力图, 滚动Sharpe, 风险归因)
- 实验报告自动生成

## [0.1.0] - 2024-12-01

### Added
- 🎉 首次发布QTMRL系统
- 基于A2C的多资产交易强化学习实现
- 数据处理模块:
  - 自动从Yahoo Finance下载股票数据
  - 9种技术指标计算 (SMA, EMA, RSI, MACD, ATR, Bollinger Bands, Ichimoku, Heikin-Ashi, SuperTrend)
  - Z-score标准化 (仅在训练集拟合)
  - 支持后复权数据
- 交易环境:
  - 多资产共享资金池
  - Factorized multi-head动作空间
  - 可配置的交易规则 (买入比例, 卖出比例, 手续费)
  - 组合级奖励函数
- 模型架构:
  - TimeCNN编码器 (1D卷积 + 全局池化)
  - Transformer编码器 (多层自注意力)
  - Multi-head Actor (每个资产独立head)
  - Critic (全局价值估计)
- A2C算法:
  - Rollout缓冲区
  - 优势函数计算
  - 策略梯度 + 熵正则 + 价值函数
  - 梯度裁剪
- 评估模块:
  - 回测功能
  - 评估指标: 总收益率, 夏普比率, 波动率, 最大回撤
  - 年化指标计算
  - 可视化: 净值曲线, 回撤曲线, 收益率分布, 动作分布
- 训练功能:
  - Checkpoint保存和恢复
  - 定期验证集评估
  - 训练指标记录 (loss, entropy, reward)
  - Wandb集成 (可选)
- 配置系统:
  - YAML配置文件
  - 两套预设配置 (完整训练 + 快速测试)
- 工具函数:
  - 随机种子设置
  - 配置加载
  - 日志记录
  - 文件IO
- 文档:
  - 详细的README (中文)
  - 快速开始指南 (QUICKSTART.md)
  - 原始设计文档 (plan.md)
  - 升级路线图 (ROADMAP.md)
- 测试:
  - 交易环境单元测试
- 其他:
  - requirements.txt
  - setup.py
  - .gitignore
  - Google Colab支持

### Technical Details
- Python 3.8+
- PyTorch 2.0+
- 支持CPU和GPU训练
- 数据范围: 2014-2024 (10年)
- 默认16只美股, 可配置
- 训练速度: ~1M步需要2-3小时 (GPU)

### Known Issues
- 某些技术指标在数据起始阶段会有NaN值 (已通过dropna处理)
- 快速测试配置的性能可能不稳定 (训练时间短)
- 尚未实现基线策略对比

---

## Version History

- **v0.1.0** (2024-12-01): 首次发布
- **v0.2.0** (计划中): 基线对比 + 早停 + 风险敏感奖励
- **v0.3.0** (计划中): 更多RL算法 + 连续动作 + 特征增强
- **v0.4.0** (计划中): 离线RL + 实时交易 + 可解释性

---

## Contributing

我们欢迎所有形式的贡献！请查看 [ROADMAP.md](ROADMAP.md) 了解计划中的功能。

提交贡献前请确保：
1. 代码通过所有测试
2. 添加必要的文档
3. 遵循项目代码风格
4. 在CHANGELOG中记录你的更改
