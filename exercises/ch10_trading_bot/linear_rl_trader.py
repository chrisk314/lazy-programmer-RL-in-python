"""Linear RL trading bot script."""

import typing as _t

import numpy as np
from numpy import random as npr
from sklearn.preprocessing import StandardScaler


TRAIN_CSV: str = "aapl_msi_sbux.csv"  # Training data of stock daily prices.
SAMPLE_EPISODES: int = 1000  # Number of episodes to sample for scaler.
LEARNING_RATE: float = 0.01  # Learning rate for SGD.
MOMENTUM: float = 0.90  # Momentum for SGD.


class Env:
    """`Env` represents an environment for an RL agent.

    Implements OpenAI Gym `gym.Env` interface.
    """

    pass


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
        self.b + +self.vb
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


def load_training_data(fname: str = TRAIN_CSV) -> np.ndarray:
    """Returns numpy array with training data."""
    return np.loadtxt(f"./data/{fname}", delimiter=",", skiprows=1)


def get_state_scaler(env: Env, n_epsd: int = SAMPLE_EPISODES) -> StandardScaler:
    """Returns scaler fit to sample from the state space."""
    samples = []
    for _ in range(n_epsd):
        s = env.reset()
        samples += [s]
        done = False
        while not done:
            a = npr.choice(env.action_space)
            s, _, done, _ = env.step(a)
            samples += [s]
    return StandardScaler().fit(samples)
