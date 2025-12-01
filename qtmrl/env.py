"""多资产交易环境"""
import numpy as np
from typing import Dict, Tuple, Optional
from enum import IntEnum


class Action(IntEnum):
    """动作枚举"""

    SELL = 0
    HOLD = 1
    BUY = 2


class TradingEnv:
    """多资产交易环境（共享资金池）"""

    def __init__(
        self,
        X: np.ndarray,  # [T, N, F] 特征张量
        Close: np.ndarray,  # [T, N] 收盘价张量
        dates: np.ndarray,  # [T] 日期数组
        window: int = 20,
        initial_cash: float = 100000.0,
        fee_rate: float = 0.0005,
        buy_pct: float = 0.20,
        sell_pct: float = 0.50,
    ):
        """初始化交易环境

        Args:
            X: 特征张量 [T, N, F]
            Close: 收盘价张量 [T, N]
            dates: 日期数组 [T]
            window: 状态窗口长度
            initial_cash: 初始资金
            fee_rate: 手续费率（单边）
            buy_pct: 买入时使用的现金比例
            sell_pct: 卖出时卖出的持仓比例
        """
        self.X = X
        self.Close = Close
        self.dates = dates
        self.window = window
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate
        self.buy_pct = buy_pct
        self.sell_pct = sell_pct

        # 数据维度
        self.T, self.N, self.F = X.shape

        # 状态变量
        self.current_step = 0
        self.cash = 0.0
        self.positions = np.zeros(self.N, dtype=np.float32)  # 持仓股数
        self.portfolio_value = 0.0

        # 历史记录
        self.portfolio_values = []
        self.actions_history = []

    def reset(self) -> Dict:
        """重置环境

        Returns:
            初始状态字典
        """
        self.current_step = self.window - 1  # 从第window-1步开始
        self.cash = self.initial_cash
        self.positions = np.zeros(self.N, dtype=np.float32)
        self.portfolio_value = self.initial_cash

        self.portfolio_values = [self.initial_cash]
        self.actions_history = []

        return self._get_state()

    def _get_state(self) -> Dict:
        """获取当前状态

        Returns:
            状态字典，包含:
            - features: [W, N, F] 历史特征窗口
            - positions: [N] 当前持仓（归一化）
            - cash: 现金（归一化）
        """
        # 特征窗口
        start_idx = self.current_step - self.window + 1
        end_idx = self.current_step + 1
        
        # Handle case where start_idx is negative (near beginning of episode)
        if start_idx < 0:
            # Pad with zeros at the beginning
            valid_start = 0
            valid_features = self.X[valid_start:end_idx].copy()  # [actual_W, N, F]
            
            # Calculate padding needed
            pad_length = -start_idx
            pad_shape = (pad_length, self.N, self.F)
            padding = np.zeros(pad_shape, dtype=np.float32)
            
            # Concatenate padding and valid features
            features = np.concatenate([padding, valid_features], axis=0)  # [W, N, F]
        else:
            features = self.X[start_idx:end_idx].copy()  # [W, N, F]

        # 归一化持仓和现金
        current_prices = self.Close[self.current_step]
        position_values = self.positions * current_prices
        total_value = self.cash + position_values.sum()

        # 持仓比例
        if total_value > 0:
            positions_norm = position_values / total_value
            cash_norm = self.cash / total_value
        else:
            positions_norm = np.zeros(self.N, dtype=np.float32)
            cash_norm = 1.0

        return {
            "features": features.astype(np.float32),
            "positions": positions_norm.astype(np.float32),
            "cash": np.array([cash_norm], dtype=np.float32),
        }

    def step(self, actions: np.ndarray) -> Tuple[Dict, float, bool, Dict]:
        """执行一步交易

        Args:
            actions: 动作数组 [N]，每个元素为 0(SELL)/1(HOLD)/2(BUY)

        Returns:
            (next_state, reward, done, info)
        """
        assert len(actions) == self.N, f"动作数量应为{self.N}，但得到{len(actions)}"

        current_prices = self.Close[self.current_step]

        # 执行交易
        for i in range(self.N):
            action = actions[i]
            price = current_prices[i]

            if price <= 0:  # 价格无效，跳过
                continue

            if action == Action.BUY:
                # 买入：使用 buy_pct * cash
                amount = self.cash * self.buy_pct
                if amount > 0:
                    # 计算手续费
                    fee = amount * self.fee_rate
                    # 可用于购买的金额
                    net_amount = amount - fee
                    # 购买股数
                    shares = net_amount / price

                    self.positions[i] += shares
                    self.cash -= amount

            elif action == Action.SELL:
                # 卖出：卖出 sell_pct * 持仓
                shares_to_sell = self.positions[i] * self.sell_pct
                if shares_to_sell > 0:
                    # 卖出金额
                    amount = shares_to_sell * price
                    # 计算手续费
                    fee = amount * self.fee_rate
                    # 净收入
                    net_amount = amount - fee

                    self.positions[i] -= shares_to_sell
                    self.cash += net_amount

        # 计算当前组合价值
        next_step = self.current_step + 1
        if next_step < self.T:
            next_prices = self.Close[next_step]
            position_values = self.positions * next_prices
            new_portfolio_value = self.cash + position_values.sum()
        else:
            new_portfolio_value = self.portfolio_value

        # 计算奖励（收益率）
        if self.portfolio_value > 0:
            reward = (new_portfolio_value / self.portfolio_value) - 1.0
        else:
            reward = 0.0

        # 更新状态
        self.portfolio_value = new_portfolio_value
        self.portfolio_values.append(self.portfolio_value)
        self.actions_history.append(actions.copy())
        self.current_step = next_step

        # 检查是否结束
        done = self.current_step >= self.T - 1

        # 获取下一个状态
        if not done:
            next_state = self._get_state()
        else:
            next_state = None

        # 信息
        info = {
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "positions": self.positions.copy(),
            "date": str(self.dates[self.current_step - 1]) if self.current_step > 0 else None,
        }

        return next_state, reward, done, info

    def get_portfolio_values(self) -> np.ndarray:
        """获取组合价值历史

        Returns:
            组合价值数组
        """
        return np.array(self.portfolio_values)

    def get_returns(self) -> np.ndarray:
        """获取收益率序列

        Returns:
            收益率数组
        """
        pv = self.get_portfolio_values()
        returns = np.diff(pv) / pv[:-1]
        return returns
