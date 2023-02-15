"""This script trains and runs an RL agent to solve the cart pole control problem.

The state and action space is treated as continuous split into bins. An SGDRegressor
implemented as a single layer tensorflow model is used.
"""
import sys
import typing as _t

import cart_pole_rbf
from cart_pole_rbf import ALPHA, Env, FeatureTransformer
import numpy as np
import tensorflow as tf


# class SGDRegressor:
#     def __init__(
#         self, dims: _t.Tuple[int, ...], learning_rate: float = cart_pole_rbf.ALPHA
#     ) -> None:
#         self.learning_rate = learning_rate

#         self.w = tf.Variable(tf.random.normal(shape=dims), name="w")

#     def partial_fit(self, X, Y) -> None:
#         self.w += self.learning_rate * np.dot(Y - np.dot(X, self.w), X)

#     def predict(self, X) -> np.ndarray:
#         return np.dot(X, self.w)


class SGDRegressor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)

        self.predict = self.call
        self.partial_fit = lambda *args: self.fit(*args, epochs=1, batch_size=1, verbose=0)

    def call(self, x):
        return self.dense(x)


class Model(cart_pole_rbf.Model):
    def __init__(self, env: Env, transformer: FeatureTransformer, learning_rate=ALPHA) -> None:
        self._trans: cart_pole_rbf.FeatureTransformer = transformer
        self._n_actions: int = env.action_space.n
        self.models: _t.List[SGDRegressor] = []
        for _ in range(self._n_actions):
            model = SGDRegressor()
            model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=ALPHA))
            self.models += [model]


if __name__ == "__main__":
    cart_pole_rbf.Model = Model
    sys.exit(cart_pole_rbf.main())
