"""This script trains and runs an RL agent to solve the cart pole control problem.

A neural network is used to learn an approximation of the optimal policy. The
policy gradient method is used to update the model.
"""

import sys
import typing as _t

from gym import Env, make
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from ..ch3_basic_RL.cart_pole_bins import plot_running_average, plot_total_rewards


ALPHA: float = 1.0e-3  # Learning rate.
GAMMA: float = 0.99  # Discount factor.
MAX_ITERS: int = 2000
NUM_EPISODES: int = 1000
CART_POLE_MAX_ITERS: int = 199


class PolicyModel(tf.keras.Model):
    def __init__(
        self, d_in: int, d_out: int, layer_sizes: _t.List[int], learning_rate: float = ALPHA
    ) -> None:
        super().__init__()
        self._layers: _t.List[tf.keras.Layer] = []
        for d in layer_sizes:
            self._layers += [tf.keras.layers.Dense(d, activation=tf.nn.tanh)]
        self._layers += [tf.keras.layers.Dense(d_out, activation=tf.nn.softmax, use_bias=False)]

        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def call(self, x: np.ndarray) -> np.ndarray:
        _x = np.atleast_2d(x)
        for l in self._layers:
            _x = l(_x)
        return _x

    def sample_action(self, x: np.ndarray) -> int:
        p = self(x, training=False)[0]
        return np.random.choice(len(p), p=p.numpy())

    def partial_fit(self, X, actions, advantages):
        # TODO : Can operations within the tape be recorded in vectorised form?
        #      : Or, should they take place sequentially, i.e. in a for loop over time steps?
        with tf.GradientTape() as tape:
            #     action_probs = self(X)  # TODO : These could have been saved during MC run...
            #     # TODO : This valid numpy indexing doesn't work in TensorFlow. How to best vectorise the for loop and does it matter?
            #     action_probs_sel = action_probs[np.arange(len(actions)), actions]
            #     log_action_probs_sel = np.log(action_probs_sel)
            #     actor_losses = log_action_probs_sel * advantages
            #     loss = np.sum(actor_losses)
            losses = []
            action_probs = self(X)
            for ap, a, adv in zip(action_probs, actions, advantages):
                losses += [-tf.math.log(ap[a]) * adv]
            loss = tf.reduce_sum(losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.trainable_variables))


class ValueModel(tf.keras.Model):
    def __init__(
        self, d_in: int, d_out: int, layer_sizes: _t.List[int], learning_rate: float = ALPHA
    ) -> None:
        super().__init__()
        self._layers: _t.List[tf.keras.Layer] = []
        for d in layer_sizes:
            self._layers += [tf.keras.layers.Dense(d, activation=tf.nn.tanh)]
        self._layers += [tf.keras.layers.Dense(d_out)]

        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def call(self, x: np.ndarray) -> np.ndarray:
        _x = np.atleast_2d(x)
        for l in self._layers:
            _x = l(_x)
        return _x

    def partial_fit(self, X, Y):
        # TODO : Can operations within the tape be recorded in vectorised form?
        #      : Or, should they take place sequentially, i.e. in a for loop over time steps?
        with tf.GradientTape() as tape:
            values = self(X)  # TODO : These could have been saved during MC run...
            loss = tf.reduce_sum((Y - values) ** 2)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.trainable_variables))


def play_one_episode_td(pi: PolicyModel, v: ValueModel, env: Env, gamma: float = GAMMA) -> float:
    return NotImplemented


def play_one_episode_mc(pi: PolicyModel, v: ValueModel, env: Env, gamma: float = GAMMA) -> float:
    """Play one Monte Carlo episode."""
    # Initialise state and action.
    _s, _ = env.reset()
    _a = pi.sample_action(_s)

    s = [_s]
    a = [_a]
    r = [0.0]

    # Play MC episode until completion or max iters.
    done = False
    iters = 0
    total_reward = 0.0
    while not done and iters < MAX_ITERS:
        _s, _r, done, _, _ = env.step(_a)
        _a = pi.sample_action(_s)

        total_reward += _r

        if done:
            _r = -200

        s += [_s]
        a += [_a]
        r += [_r]

        iters += 1

    # Construct episode returns and advantages.
    G = []
    adv = []
    _G = 0.0
    while iters > 0:
        iters -= 1
        G += [_G]
        adv += [_G - v(s[iters])[0]]
        _G = r[iters] + GAMMA * _G

    G = G[::-1]
    adv = adv[::-1]

    # Update policy and value models.
    pi.partial_fit(s, a, adv)
    v.partial_fit(s, G)

    return total_reward


def play_one_episode(
    *args: _t.Any,
    method: str = "TD",
    **kwargs: _t.Any,
) -> float:
    if method == "MC":
        return play_one_episode_mc(*args, **kwargs)
    elif method == "TD":
        return play_one_episode_td(*args, **kwargs)
    else:
        raise ValueError(f"Unrecognised method: {method}.")


def main() -> int:
    # Set up environment and agent.
    env = make("CartPole-v0")

    # Create NN models
    d_in = env.observation_space.shape[0]
    d_out = env.action_space.n
    pi = PolicyModel(d_in, d_out, [10, 10])
    v = ValueModel(d_in, 1, [10])

    # Train agent.
    total_rewards = []
    for n in tqdm(range(NUM_EPISODES)):
        total_rewards += [play_one_episode(pi, v, env, method="MC")]
        if n % 100 == 0:
            print(f"Episode {n}: total reward: {total_rewards[-1]}.")

    # Plot results.
    plot_total_rewards(total_rewards)
    plot_running_average(total_rewards)

    return 0


if __name__ == "__main__":
    sys.exit(main())
