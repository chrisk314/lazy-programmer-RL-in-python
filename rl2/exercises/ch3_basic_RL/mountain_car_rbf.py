"""This script trains and runs an RL agent to solve the mountain car control problem.

The state and action space is treated as continuous split into bins. Several RBFSamplers
are stacked together to produce features from the input data.
"""

import sys
import typing as _t

from gym import Env, make
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


ALPHA: float = 1.0e-2  # Learning rate.
GAMMA: float = 0.99  # Discount factor.
EPSILON: float = 0.05  # Exploration rate.
MAX_ITERS: int = 10000
NUM_EPISODES: int = 300


# Note: gym changed from version 0.7.3 to 0.8.0
# MountainCar episode length is capped at 200 in later versions.
# This means your agent can't learn as much in the earlier episodes
# since they are no longer as long.

# SGDRegressor defaults:
# loss='squared_loss', penalty='l2', alpha=0.0001,
# l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=True,
# verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling',
# eta0=0.01, power_t=0.25, warm_start=False, average=False


class FeatureTransformer:
    def __init__(self, env: Env, n_components: int = 500) -> None:
        n_samples = 10000
        state_samples = np.array([env.observation_space.sample() for _ in range(n_samples)])
        # Scaler ensures data has 0 mean and unit variance for compatibility with ML algos..
        self.scaler = StandardScaler()
        self.scaler.fit(state_samples)

        # Used to convert a state to a featurised representation.
        # RBF kernels with different variances cover different parts of the space.
        self.featurizer = FeatureUnion(
            [
                ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
                ("rbf2", RBFSampler(gamma=1.0, n_components=n_components)),
                ("rbf3", RBFSampler(gamma=5.0, n_components=n_components)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=n_components)),
            ]
        )

        # TODO : Seems liks a hack to get dimensions. Is there a better way?
        examples = self.featurizer.fit_transform(self.scaler.transform(state_samples))
        self.dimensions = examples.shape[1]

    def transform(self, obs: np.ndarray) -> int:
        scaled_obs = self.scaler.transform(obs)
        return self.featurizer.transform(scaled_obs)


class Model:
    def __init__(
        self,
        env: Env,
        transformer: FeatureTransformer,
        learning_rate=ALPHA,
        regressor_cls: _t.Type = SGDRegressor,
    ) -> None:
        self._trans: FeatureTransformer = transformer
        self._n_actions: int = env.action_space.n
        self.models: SGDRegressor = []
        for _ in range(self._n_actions):
            model = regressor_cls(learning_rate=learning_rate)
            states = [env.reset()[0]]
            model.partial_fit(self._trans.transform(states), [0.0])
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

        # Update the model
        G = r + gamma * max(model.predict(s2))
        model.update(s, a, G)

        # Update iteration vars
        s = s2
        total_reward += r
        iters += 1

    return total_reward


def plot_cost_to_go(env: Env, estimator: Model, num_tiles: int = 20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    # both X and Y will be of shape (num_tiles, num_tiles)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
    # Z will also be of shape (num_tiles, num_tiles)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_zlabel("Cost-To-Go == -V(s)")
    ax.set_title("Cost-To-Go Function")
    fig.colorbar(surf)
    plt.show()


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
    env = make("MountainCar-v0")
    trans = FeatureTransformer(env)
    model = Model(env, trans, learning_rate="constant")
    total_rewards = []

    # Train agent.
    for n in tqdm(range(NUM_EPISODES)):
        eps = 1.0 / np.sqrt(n + 1)
        total_rewards += [play_one_episode(model, env, eps=eps)]
        if n % 20 == 0:
            print(f"Episode {n}: total reward: {total_rewards[-1]}, eps: {eps}.")

    # Plot results.
    plot_total_rewards(total_rewards)
    plot_running_average(total_rewards)
    plot_cost_to_go(env, model)

    return 0


if __name__ == "__main__":
    sys.exit(main())
