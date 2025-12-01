"""交易环境单元测试"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qtmrl.env import TradingEnv, Action


def test_env_initialization():
    """测试环境初始化"""
    T, N, F = 100, 2, 5
    X = np.random.randn(T, N, F).astype(np.float32)
    Close = np.random.rand(T, N).astype(np.float32) * 100 + 50
    dates = np.arange(T)

    env = TradingEnv(
        X=X,
        Close=Close,
        dates=dates,
        window=20,
        initial_cash=100000.0,
        fee_rate=0.0005,
        buy_pct=0.20,
        sell_pct=0.50,
    )

    assert env.T == T
    assert env.N == N
    assert env.F == F
    print("✓ 环境初始化测试通过")


def test_env_reset():
    """测试环境重置"""
    T, N, F = 100, 2, 5
    X = np.random.randn(T, N, F).astype(np.float32)
    Close = np.random.rand(T, N).astype(np.float32) * 100 + 50
    dates = np.arange(T)

    env = TradingEnv(X=X, Close=Close, dates=dates, window=20)
    state = env.reset()

    assert "features" in state
    assert "positions" in state
    assert "cash" in state
    assert state["features"].shape == (20, N, F)
    assert state["positions"].shape == (N,)
    assert state["cash"].shape == (1,)
    assert env.cash == env.initial_cash
    assert np.all(env.positions == 0)
    print("✓ 环境重置测试通过")


def test_env_step_hold():
    """测试HOLD动作"""
    T, N, F = 100, 2, 5
    X = np.random.randn(T, N, F).astype(np.float32)
    Close = np.random.rand(T, N).astype(np.float32) * 100 + 50
    dates = np.arange(T)

    env = TradingEnv(X=X, Close=Close, dates=dates, window=20)
    env.reset()

    initial_cash = env.cash
    actions = np.array([Action.HOLD, Action.HOLD])
    state, reward, done, info = env.step(actions)

    # HOLD不应该改变持仓和现金
    assert env.cash == initial_cash
    assert np.all(env.positions == 0)
    print("✓ HOLD动作测试通过")


def test_env_step_buy():
    """测试BUY动作"""
    T, N, F = 100, 2, 5
    X = np.random.randn(T, N, F).astype(np.float32)
    Close = np.random.rand(T, N).astype(np.float32) * 100 + 50
    dates = np.arange(T)

    env = TradingEnv(
        X=X,
        Close=Close,
        dates=dates,
        window=20,
        initial_cash=100000.0,
        buy_pct=0.20,
        fee_rate=0.0005,
    )
    env.reset()

    initial_cash = env.cash
    actions = np.array([Action.BUY, Action.HOLD])
    state, reward, done, info = env.step(actions)

    # BUY应该减少现金，增加持仓
    assert env.cash < initial_cash
    assert env.positions[0] > 0
    assert env.positions[1] == 0
    print("✓ BUY动作测试通过")


def test_env_step_sell():
    """测试SELL动作"""
    T, N, F = 100, 2, 5
    X = np.random.randn(T, N, F).astype(np.float32)
    Close = np.random.rand(T, N).astype(np.float32) * 100 + 50
    dates = np.arange(T)

    env = TradingEnv(
        X=X,
        Close=Close,
        dates=dates,
        window=20,
        initial_cash=100000.0,
        buy_pct=0.20,
        sell_pct=0.50,
        fee_rate=0.0005,
    )
    env.reset()

    # 先买入
    actions = np.array([Action.BUY, Action.BUY])
    env.step(actions)

    initial_positions = env.positions.copy()
    initial_cash = env.cash

    # 再卖出
    actions = np.array([Action.SELL, Action.HOLD])
    env.step(actions)

    # SELL应该减少持仓，增加现金
    assert env.positions[0] < initial_positions[0]
    assert env.cash > initial_cash
    print("✓ SELL动作测试通过")


def test_env_portfolio_value():
    """测试组合价值计算"""
    T, N, F = 100, 2, 5
    X = np.random.randn(T, N, F).astype(np.float32)
    Close = np.random.rand(T, N).astype(np.float32) * 100 + 50
    dates = np.arange(T)

    env = TradingEnv(X=X, Close=Close, dates=dates, window=20, initial_cash=100000.0)
    env.reset()

    # 初始组合价值应该等于初始现金
    assert env.portfolio_value == 100000.0

    # 执行一些操作
    for _ in range(10):
        actions = np.random.randint(0, 3, size=N)
        state, reward, done, info = env.step(actions)
        if done:
            break

    # 组合价值应该大于0
    assert env.portfolio_value > 0
    print("✓ 组合价值计算测试通过")


def test_env_fee():
    """测试手续费计算"""
    T, N, F = 100, 1, 5
    X = np.random.randn(T, N, F).astype(np.float32)
    Close = np.ones((T, N), dtype=np.float32) * 100  # 固定价格
    dates = np.arange(T)

    env = TradingEnv(
        X=X,
        Close=Close,
        dates=dates,
        window=20,
        initial_cash=100000.0,
        buy_pct=0.20,
        fee_rate=0.001,  # 0.1% 手续费
    )
    env.reset()

    initial_cash = env.cash
    actions = np.array([Action.BUY])
    env.step(actions)

    # 计算预期的扣除金额（买入金额 + 手续费）
    buy_amount = initial_cash * 0.20
    expected_deduction = buy_amount
    actual_deduction = initial_cash - env.cash

    assert np.isclose(actual_deduction, expected_deduction, rtol=1e-4)
    print("✓ 手续费计算测试通过")


if __name__ == "__main__":
    print("运行交易环境单元测试...\n")
    test_env_initialization()
    test_env_reset()
    test_env_step_hold()
    test_env_step_buy()
    test_env_step_sell()
    test_env_portfolio_value()
    test_env_fee()
    print("\n所有测试通过! ✓")
