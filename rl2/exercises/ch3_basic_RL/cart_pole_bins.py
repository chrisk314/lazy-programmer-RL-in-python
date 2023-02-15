"""This script trains and runs an RL agent to solve the cart pole control problem.

The state and action space is treated as continuous split into bins.
"""

import sys
import typing as _t

from gym import Env, make
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


ALPHA: float = 1.0e-1  # Learning rate.
GAMMA: float = 0.90  # Discount factor.
EPSILON: float = 0.05  # Exploration rate.
MAX_ITERS: int = 10000
NUM_EPISODES: int = 10000

CART_POLE_MAX_ITERS: int = 199
STATES_DISCRETISATION: int = 10  # Number of bins to split continuous space over


class FeatureTransformer:
    # Ideally the bin parameters should be determined through EDA by taking
    # samples from the environment and analysing the distribution of values.
    # A further improvement would be to have inhomogeneous bin sized chosen
    # to give uniform probability of observations being assigned to the bins.

    @staticmethod
    def _to_bin(values: float, bins: np.ndarray) -> int:
        return np.digitize([values], bins)[0]

    def _build_state(self, bpc: int, bvc: int, bpp: int, bvp: int) -> int:
        return ((bvp * self._nbpp + bpp) * self._nbvc + bvc) * self._nbpc + bpc

    def __init__(self) -> None:
        self._nbpc: int = STATES_DISCRETISATION  # n. bins pos. cart
        self._nbvc: int = STATES_DISCRETISATION  # n. bins vel. cart
        self._nbpp: int = STATES_DISCRETISATION  # n. bins pos. pole
        self._nbvp: int = STATES_DISCRETISATION  # n. bins vel. pole
        self._bins_pos_cart: np.ndarray = np.linspace(-2.4, 2.4, self._nbpc - 1)
        self._bins_vel_cart: np.ndarray = np.linspace(-2.0, 2.0, self._nbvc - 1)
        self._bins_pos_pole: np.ndarray = np.linspace(-0.4, 0.4, self._nbpp - 1)
        self._bins_vel_pole: np.ndarray = np.linspace(-0.9, 0.9, self._nbvp - 1)

    def transform(self, obs: np.ndarray) -> int:
        pos_cart, vel_cart, pos_pole, vel_pole = obs
        return self._build_state(
            self._to_bin(pos_cart, self._bins_pos_cart),
            self._to_bin(vel_cart, self._bins_vel_cart),
            self._to_bin(pos_pole, self._bins_pos_pole),
            self._to_bin(vel_pole, self._bins_vel_pole),
        )


class Model:
    def __init__(self, env: Env, transformer: FeatureTransformer) -> None:
        self._trans: FeatureTransformer = transformer
        # TODO : `FeatureTransformer` should expose size of state space.
        self._n_states: int = STATES_DISCRETISATION ** env.observation_space.shape[0]
        self._n_actions: int = env.action_space.n
        self.Q: np.ndarray = np.random.uniform(
            low=-1.0, high=1.0, size=(self._n_states, self._n_actions)
        )

    def predict(self, s: int) -> np.ndarray:
        """Returns array of action values for given state, `s`."""
        x = self._trans.transform(s)
        return self.Q[x]

    def update(self, s: int, a: int, G: float) -> None:
        """Updates `Q` for state `s` and action `a` based on return `G`."""
        x = self._trans.transform(s)
        self.Q[x, a] += ALPHA * (G - self.Q[x, a])

    def sample_action(self, s: int, eps: float = EPSILON) -> int:
        if np.random.random() < eps:
            return np.random.choice(self._n_actions)
        return np.argmax(self.predict(s))


def play_one_episode(model: Model, env: Env, eps: float = EPSILON, gamma: float = GAMMA) -> float:
    s, _ = env.reset()
    done = False
    iters = 0
    total_reward = 0.0
    while not done and iters < MAX_ITERS:
        # Choose action and take a step
        a = model.sample_action(s, eps=eps)
        s2, r, done, _, _ = env.step(a)

        total_reward += r

        # Apply penalty for dropping the pole
        if done and iters < CART_POLE_MAX_ITERS:
            r = -300

        # Update the model
        G = r + gamma * max(model.predict(s2))
        model.update(s, a, G)

        # Update iteration vars
        s = s2
        iters += 1

    return total_reward


def plot_total_rewards(total_rewards: _t.List[float]) -> None:
    plt.plot(total_rewards)
    plt.title("Total rewards")
    plt.show()


def plot_running_average(total_rewards: _t.List[float]) -> None:
    r_cum: np.ndarray = np.cumsum(total_rewards)
    r_running_avg: np.ndarray = r_cum / np.arange(1, len(total_rewards) + 1)
    plt.plot(r_running_avg)
    plt.title("Running average total rewards")
    plt.show()


def main() -> int:
    # Set up environment and agent.
    env = make("CartPole-v0")
    trans = FeatureTransformer()
    model = Model(env, trans)
    total_rewards = []

    # Train agent.
    for n in tqdm(range(NUM_EPISODES)):
        eps = 1.0 / np.sqrt(n + 1)
        total_rewards += [play_one_episode(model, env, eps=eps)]
        if n % 100 == 0:
            print(f"Episode {n}: total reward: {total_rewards[-1]}, eps: {eps}.")

    # Plot results.
    plot_total_rewards(total_rewards)
    plot_running_average(total_rewards)

    return 0


if __name__ == "__main__":
    sys.exit(main())
