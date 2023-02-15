"""This script trains and runs an RL agent to solve the cart pole control problem.

The state and action space is treated as continuous split into bins. Several RBFSamplers
are stacked together to produce features from the input data.
"""

import sys
import typing as _t

from gym import Env, make
from matplotlib import pyplot as plt
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


ALPHA: float = 1.0e-1  # Learning rate.
GAMMA: float = 0.99  # Discount factor.
EPSILON: float = 0.05  # Exploration rate.
MAX_ITERS: int = 2000
NUM_EPISODES: int = 1000

CART_POLE_MAX_ITERS: int = 199
STATES_DISCRETISATION: int = 10  # Number of bins to split continuous space over


class FeatureTransformer:
    def __init__(self, env: Env, n_components: int = 500) -> None:
        n_samples = 20000
        # Generate uniformly distributed samples around the
        state_samples = np.random.random((n_samples, env.observation_space.shape[0])) * 4.0 - 2.0
        # Scaler ensures data has 0 mean and unit variance for compatibility with ML algos..
        self.scaler = StandardScaler()
        self.scaler.fit(state_samples)

        # Used to convert a state to a featurised representation.
        # RBF kernels with different variances cover different parts of the space.
        self.featurizer = FeatureUnion(
            [
                ("rbf1", RBFSampler(gamma=0.05, n_components=n_components)),
                ("rbf2", RBFSampler(gamma=1.0, n_components=n_components)),
                ("rbf3", RBFSampler(gamma=0.5, n_components=n_components)),
                ("rbf4", RBFSampler(gamma=0.1, n_components=n_components)),
            ]
        )

        # TODO : Seems liks a hack to get dimensions. Is there a better way?
        examples = self.featurizer.fit_transform(self.scaler.transform(state_samples))
        self.dimensions = examples.shape[1]

    def transform(self, obs: np.ndarray) -> int:
        scaled_obs = self.scaler.transform(obs)
        return self.featurizer.transform(scaled_obs)


class SGDRegressor:
    def __init__(self, dims: _t.Tuple[int, ...], learning_rate: float = ALPHA) -> None:
        self.w = np.random.randn(dims) / np.sqrt(np.product(dims))
        self.learning_rate = learning_rate

    def partial_fit(self, X, Y) -> None:
        self.w += self.learning_rate * np.dot(Y - np.dot(X, self.w), X)

    def predict(self, X) -> np.ndarray:
        return np.dot(X, self.w)


class Model:
    def __init__(self, env: Env, transformer: FeatureTransformer, learning_rate=ALPHA) -> None:
        self._trans: FeatureTransformer = transformer
        self._n_actions: int = env.action_space.n
        self.models: _t.List[SGDRegressor] = []
        for _ in range(self._n_actions):
            model = SGDRegressor(self._trans.dimensions, learning_rate=learning_rate)
            # states = [env.reset()[0]]
            # model.partial_fit(self._trans.transform(states), [0.0])
            self.models += [model]

    def predict(self, s: int) -> np.ndarray:
        """Returns array of action values for given state, `s`."""
        X = self._trans.transform([s])
        return np.array([m.predict(X)[0] for m in self.models])

    def update(self, s: int, a: int, G: float) -> None:
        """Updates `Q` for state `s` and action `a` based on return `G`."""
        X = self._trans.transform([s])
        self.models[a].partial_fit(X, [G])

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
    trans = FeatureTransformer(env)
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
