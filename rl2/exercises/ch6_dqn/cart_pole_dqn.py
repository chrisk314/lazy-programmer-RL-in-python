"""This script trains and runs an RL agent to solve the cart pole control problem.

A deep neural network is used to learn an approximation of the optimal policy.
Deep Q learning with experience replay is used to update the model.
"""

from __future__ import annotations

from collections import deque
import sys
import typing as _t

from gym import Env, make
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from ..ch3_basic_RL.cart_pole_bins import plot_running_average, plot_total_rewards


ALPHA: float = 1.0e-2  # Learning rate.
GAMMA: float = 0.99  # Discount factor.
EPSILON: float = 0.05  # Exploration rate.
MAX_ITERS: int = 2000
NUM_EPISODES: int = 1000
CART_POLE_MAX_ITERS: int = 200

# See D. Silver RL lectures Lecture 6: Value Function Approximation for comparison of different lambdas
LAMBDA: float = 0.7
TARGET_UPDATE_ITERS: int = 50  # Iters between updates to target model
MINI_BATCH_SIZE: int = 32
MIN_EXPERIENCE: int = 100
MAX_EXPERIENCE: int = 10000

# TODO : Improve architecture to avoid global var.
# Global iteration counter for target network update.
GLOBAL_ITERS: int = 0


class DQN(tf.keras.Model):
    def __init__(
        self, d_in: int, d_out: int, layer_sizes: _t.List[int], learning_rate: float = ALPHA
    ) -> None:
        super().__init__()
        self._n_actions = d_out
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

    def sample_action(self, s: int, eps: float = EPSILON) -> int:
        if np.random.random() < eps:
            return np.random.choice(self._n_actions)
        _s = np.atleast_2d(s)
        return np.argmax(self(_s)[0])

    def partial_fit(self, X, actions, Y):
        with tf.GradientTape() as tape:
            action_values = self(X)
            selected_action_values = tf.reduce_sum(
                action_values * tf.one_hot(actions, self._n_actions), axis=1
            )
            loss = tf.reduce_sum(tf.square(Y - selected_action_values))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.trainable_variables))

    def train_from_buffer(self, buffer: _t.Deque, Q_t: DQN, gamma: float = GAMMA) -> None:
        if len(buffer) < MIN_EXPERIENCE:
            # Skip if not enough experience in the buffer yet.
            return
        # Sample a random batch without replacement.
        batch = [
            buffer[i] for i in np.random.choice(len(buffer), size=MINI_BATCH_SIZE, replace=False)
        ]
        s, a, r, s2, done = zip(*batch)

        # Calculate stable returns from target network.
        G = r + np.where(~np.array(done), gamma * np.max(Q_t(s2), axis=1), 0.0)

        # Train the model on the batch.
        self.partial_fit(s, a, G)

    def copy_from(self, other) -> None:
        """Copies network parameters from another `DQN`."""
        self.set_weights(other.get_weights())


def play_one_episode_td(
    Q: DQN,
    Q_t: DQN,
    env: Env,
    buffer: _t.Deque,
    gamma: float = GAMMA,
    eps: float = EPSILON,
) -> float:
    global GLOBAL_ITERS

    s, _ = env.reset()
    done = False
    iters = 0
    total_reward = 0.0

    # TODO : Figure out the proper way to initialise the state of `Q_t` and avoid the error message below.
    # ValueError: You called `set_weights(weights)` on layer "dqn_1" with a weight list of length 6, but the layer was expecting 0 weights
    if GLOBAL_ITERS == 0:
        Q_t(s)  # Call `Q_t` to initialise state... there must be a better way...

    while not done and iters < MAX_ITERS:
        # Choose action using Q network and take a step
        a = Q.sample_action(s, eps=eps)
        s2, r, done, _, _ = env.step(a)

        total_reward += r

        if done and iters < CART_POLE_MAX_ITERS:
            r = -200

        # Add the experience to the replay buffer
        buffer.append((s, a, r, s2, done))

        # Train the agent from the experience buffer
        Q.train_from_buffer(buffer, Q_t, gamma=gamma)

        # Update iteration vars
        s = s2
        iters += 1
        GLOBAL_ITERS += 1

        # Update the target network after some iters
        if GLOBAL_ITERS % TARGET_UPDATE_ITERS == 0:
            Q_t.copy_from(Q)

    return total_reward


def play_one_episode(
    *args: _t.Any,
    method: str = "TD",
    **kwargs: _t.Any,
) -> float:
    if method == "TD":
        return play_one_episode_td(*args, **kwargs)
    # elif method == "MC":
    #     return play_one_episode_mc(*args, **kwargs)
    else:
        raise ValueError(f"Unrecognised method: {method}.")


def main() -> int:
    # Set up environment and agent.
    env = make("CartPole-v0")

    # Create NN models
    d_in = env.observation_space.shape[0]
    d_out = env.action_space.n
    Q = DQN(d_in, d_out, [200, 200])  # Q network to train
    Q_t = DQN(d_in, d_out, [200, 200])  # Target Q network to stabilise gradients

    # Create replay buffer with max length.
    buffer = deque(maxlen=MAX_EXPERIENCE)

    # Train agent.
    total_rewards = []
    for n in tqdm(range(NUM_EPISODES)):
        eps = 1.0 / np.sqrt(n + 1)
        total_rewards += [play_one_episode(Q, Q_t, env, buffer, method="TD", eps=eps)]
        if n == 0 or (n + 1) % 100 == 0:
            print(f"Episode {n}: total reward: {total_rewards[-1]}.")
            # Q.save("cart-pole-dqn.tf")

    # Plot results.
    plot_total_rewards(total_rewards)
    plot_running_average(total_rewards)

    return 0


if __name__ == "__main__":
    sys.exit(main())
