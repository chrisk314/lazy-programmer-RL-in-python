"""Linear RL trading bot script."""
from __future__ import annotations

from itertools import product
from pathlib import Path
import pickle
import typing as _t

import click
from gym import Env
from matplotlib import pyplot as plt
import numpy as np
from numpy import random as npr
import pandas as pd
from sklearn.preprocessing import StandardScaler


TRAIN_CSV: str = "aapl_msi_sbux.csv"  # Training data of stock daily prices.
SAMPLE_EPISODES: int = 1000  # Number of episodes to sample for scaler.
LEARNING_RATE: float = 0.01  # Learning rate for SGD.
MOMENTUM: float = 0.90  # Momentum for SGD.
INITIAL_INVESTMENT = 10000


class HistoricMultiStockEnv(Env):
    """Environment for RL agent trading multiple stocks.

    Implements the gym.Env interface.

    State: vector of size 7 (2 * n_stocks + 1)
        - (count stock 1...N, $ stock 1...N, cash)
    Actions: categorical variable with size 27=3^3 (n_stocks^3)
        - For each stock 0: sell, 1: buy, 2: hold
    """

    def __init__(
        self, historic_prices: pd.DataFrame, initial_investment: float = INITIAL_INVESTMENT
    ) -> None:
        self.historic_prices: pd.DataFrame = historic_prices
        self.n_steps: int = self.historic_prices.shape[0]
        self.n_stocks: int = self.historic_prices.shape[1]
        self.initial_investment: float = initial_investment
        self.action_space: np.ndarray = np.arange(3**self.n_stocks)
        self.action_list: _t.List = [list(x) for x in product([0, 1, 2], repeat=self.n_stocks)]
        self.state_dim: int = self.n_stocks * 2 + 1

        # Initialise state data
        self.cur_step: int = 0
        self.stock_owned: np.ndarray = np.zeros(self.n_stocks)
        self.stock_price: np.ndarray = self.historic_prices[self.cur_step]
        self.cash: float = self.initial_investment

    def _get_value(self) -> float:
        return np.dot(self.stock_owned, self.stock_price) + self.cash

    def _get_obs(self) -> np.ndarray:
        return np.array([*self.stock_owned, *self.stock_price, self.cash])

    def reset(self) -> _t.Tuple[np.ndarray, _t.Dict]:
        self.cur_step = 0
        self.stock_owned = np.zeros(self.n_stocks)
        self.stock_price = self.historic_prices[self.cur_step]
        self.cash = self.initial_investment
        return self._get_obs(), {"cur_val": self.initial_investment}

    def _trade(self, action: int) -> None:
        action_vec: np.ndarray = self.action_list[action]
        to_buy: _t.List = []
        for stock in range(self.n_stocks):
            if action_vec[stock] == 0:
                self.cash = self.stock_owned[stock] * self.stock_price[stock]
                self.stock_owned[stock] = 0
            elif action_vec[stock] == 1:
                to_buy += [stock]
        min_price = min([self.stock_price[stock] for stock in to_buy]) if to_buy else None
        while to_buy and self.cash > min_price:
            for stock in to_buy:
                if self.stock_price[stock] < self.cash:
                    self.stock_owned[stock] += 1
                    self.cash -= self.stock_price[stock]

    def step(self, action: int) -> _t.Tuple:
        v_prev: float = self._get_value()
        self.cur_step += 1
        self.stock_price = self.historic_prices[self.cur_step]
        self._trade(action)
        v_cur = self._get_value()
        # Reward is equal to change in the portfolio value
        r = v_cur - v_prev
        done = self.cur_step == self.n_steps - 1
        trunc = False  # Indicates if episode has been truncated
        info = {"cur_val": v_cur}
        return self._get_obs(), r, done, trunc, info


class LinearModel:
    """Linear regression model."""

    def __init__(self, d_in: int, d_out: int) -> None:
        self.W: np.ndarray = npr.random((d_in, d_out)) / np.sqrt(d_in)  # Weight matrix.
        self.b: np.ndarray = np.zeros(d_out)  # Bias terms.

        self.vW: float = 0.0  # Momentum term for weights.
        self.vb: float = 0.0  # Momentum term for biases.

        self.losses: _t.List[float] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns prediction for states in matrix `X`."""
        return np.dot(X, self.W) + self.b

    def sgd(
        self, X: np.ndarray, Y: np.ndarray, lr: float = LEARNING_RATE, mom: float = MOMENTUM
    ) -> None:
        """Take one step of SGD with momentum."""
        # TODO : Check understanding of SGD theory.
        _pre: float = 2.0 / np.prod(Y.shape)  # Prefactor for updates from partial deriv.
        Y_hat: np.ndarray = self.predict(X)
        Y_err: np.ndarray = Y_hat - Y
        gW: np.ndarray = _pre * np.dot(X.T, Y_err)  # Gradient wrt weights.
        gb: np.ndarray = _pre * np.sum(Y_err, axis=0)  # Gradient wrt biases.
        self.vW = mom * self.vW - lr * gW  # Update momentum for weights.
        self.vb = mom * self.vb - lr * gb  # Update momentum for biases.
        self.W += self.vW
        self.b += self.vb
        self.losses += [np.mean(Y_err**2)]  # Track MSE.

    def load_weights(self, path: str) -> None:
        """Loads model weights from `path`."""
        npz = np.load(path)
        self.W = npz["W"]
        self.b = npz["b"]

    def save_weights(self, path: str) -> None:
        """Saves model weights to `path`."""
        # TODO : Check details of file format.
        np.savez(path, W=self.W, b=self.b)


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        gamma: float = 0.95,
        eps: float = 0.1,
        eps_min: float = 0.01,
        eps_decay: float = 0.99,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.model = LinearModel(self.state_size, self.action_size)

    def act(self, state: np.ndarray) -> int:
        if npr.random() <= self.eps:
            return npr.choice(self.action_size)
        else:
            action_values = self.model.predict(state)
            return np.argmax(action_values[0])

    def train(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> None:
        target = reward
        if not done:
            target += self.gamma * np.max(self.model.predict(next_state), axis=1)
        target_full = self.model.predict(state)
        target_full[0, action] = target
        self.model.sgd(state, target_full)
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

    def load(self, path: Path) -> None:
        self.model.load_weights(path)

    def save(self, path: Path) -> None:
        self.model.save_weights(path)


def play_episode(
    agent: DQNAgent, env: Env, scaler: StandardScaler, is_train: bool = False
) -> _t.Dict:
    s, _ = env.reset()
    s = np.array(scaler.transform([s]))
    while True:
        a = agent.act(s)
        s2, r, done, trunc, info = env.step(a)
        s2 = np.array(scaler.transform([s2]))
        if is_train:
            agent.train(s, a, r, s2, done)
        s = s2
        if done or trunc:
            break
    return info


def load_training_data(fname: str = TRAIN_CSV) -> np.ndarray:
    """Returns numpy array with training data."""
    return np.loadtxt(f"./data/{fname}", delimiter=",", skiprows=1)


def get_state_scaler(env: Env, n_epsd: int = SAMPLE_EPISODES) -> StandardScaler:
    """Returns scaler fit to sample from the state space."""
    samples = []
    for _ in range(n_epsd):
        s, _ = env.reset()
        samples += [s]
        done = False
        while not done:
            a = npr.choice(env.action_space)
            s, _, done, _, _ = env.step(a)
            samples += [s]
    return StandardScaler().fit(samples)


@click.command()
@click.option("--train", "-t", is_flag=True, help="Performs model training if set.")
def main(train: bool) -> int:
    models_dir = Path("./data/linear_rl_trader_models")
    rewards_dir = Path("./data/linear_rl_trader_rewards")
    models_dir.mkdir(parents=True, exist_ok=True)
    rewards_dir.mkdir(parents=True, exist_ok=True)

    n_episodes = 2000
    batch_size = 32
    initial_investment = 1e5

    data = load_training_data()
    n_steps, n_stocks = data.shape

    n_steps_train = n_steps // 2
    data_train, data_test = data[:n_steps_train], data[n_steps_train:]

    if train:
        env = HistoricMultiStockEnv(data_train, initial_investment=initial_investment)
        agent = DQNAgent(env.state_dim, len(env.action_space))
        scaler = get_state_scaler(env)
    else:
        # Running in test mode so load the `agent` and `scaler`.
        env = HistoricMultiStockEnv(data_test, initial_investment=initial_investment)
        agent = DQNAgent(env.state_dim, len(env.action_space))
        agent.load(models_dir / "linear.npz")
        agent.eps = 0.01
        with (models_dir / "scaler.pkl").open("rb") as f:
            scaler = pickle.load(f)

    portfolio_value = []
    for _ in range(n_episodes):
        info = play_episode(agent, env, scaler, is_train=train)
        portfolio_value += [info["cur_val"]]

    if train:
        # Running in train mode so save the `agent` and `scaler`.
        agent.save(models_dir / "linear.npz")
        with (models_dir / "scaler.pkl").open("rb") as f:
            pickle.dump(scaler, f)

        # Plot the agent performance
        plt.plot(agent.model.losses)
        plt.show()

    np.save(rewards_dir / ("train.npy" if train else "test.npy"), portfolio_value)


if __name__ == "__main__":
    main()
