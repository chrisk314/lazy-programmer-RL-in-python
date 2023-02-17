"""This script trains and runs an RL agent to solve the mountain car control problem.

The state and action space is treated as continuous split into bins. Several RBFSamplers
are stacked together to produce features from the input data.

The TD lambda method is used to evaluate the return, G, and update the model with eligibility
traces.
"""

from collections import deque
import sys
import typing as _t

from gym import Env, make
import numpy as np
from tqdm import tqdm
from ..ch3_basic_RL.mountain_car_rbf import (
    ALPHA,
    EPSILON,
    FeatureTransformer,
    GAMMA,
    MAX_ITERS,
    NUM_EPISODES,
    plot_cost_to_go,
    plot_running_average,
    plot_total_rewards,
)


# See D. Silver RL lectures Lecture 6: Value Function Approximation for comparison of different lambdas
LAMBDA: float = 0.7


class BaseModel:
    def __init__(self, dims: _t.Tuple[int, ...]) -> None:
        self.w = np.random.randn(dims) / np.sqrt(np.product(dims))

    def partial_fit(
        self, X: np.ndarray, Y: np.ndarray, eligibility: np.ndarray, learning_rate: float = ALPHA
    ) -> None:
        self.w += learning_rate * (Y - np.dot(X, self.w)) * eligibility

    def predict(self, X) -> np.ndarray:
        return np.dot(X, self.w)


class Model:
    def __init__(self, env: Env, transformer: FeatureTransformer) -> None:
        self._trans: FeatureTransformer = transformer
        self._n_actions: int = env.action_space.n
        self.models: _t.List[BaseModel] = []
        self.eligibilities = np.zeros((env.action_space.n, self._trans.dimensions))
        for _ in range(self._n_actions):
            model = BaseModel(self._trans.dimensions)
            self.models += [model]

    def predict(self, s: int) -> np.ndarray:
        """Returns array of action values for given state, `s`."""
        X = self._trans.transform([s])
        return np.array([m.predict(X)[0] for m in self.models])

    def update(
        self,
        s: int,
        a: int,
        G: float,
        gamma: float = GAMMA,
        _lambda: float = LAMBDA,
        learning_rate: float = ALPHA,
    ) -> None:
        """Updates `Q` for state `s` and action `a` based on return `G`."""
        X = self._trans.transform([s])
        # Discount the old eligibility values
        self.eligibilities *= gamma * _lambda
        # See D. Silver RL lectures Lecture 6: Value Function Approximation for update explanation
        self.eligibilities[a] += X[0]
        self.models[a].partial_fit(X[0], G, self.eligibilities[a], learning_rate=learning_rate)

    def sample_action(self, s: int, eps: float = EPSILON) -> int:
        if np.random.random() < eps:
            return np.random.choice(self._n_actions)
        return np.argmax(self.predict(s))


def play_one_episode(
    model: Model, env: Env, eps: float = EPSILON, gamma: float = GAMMA, _lambda: float = LAMBDA
) -> float:
    s, _ = env.reset()
    done = False
    iters = 0
    total_reward = 0.0

    while not done and iters < MAX_ITERS:
        # Choose action and take a step
        a = model.sample_action(s, eps=eps)
        s2, r, done, _, _ = env.step(a)

        # Update the model with N-steps of data
        G = r + gamma * max(model.predict(s2))
        model.update(s, a, G, gamma=gamma, _lambda=_lambda)

        # Update iteration vars
        s = s2
        total_reward += r
        iters += 1

    return total_reward


def main() -> int:
    # Set up environment and agent.
    env = make("MountainCar-v0")
    trans = FeatureTransformer(env)
    model = Model(env, trans)
    total_rewards = []

    # Train agent.
    for n in tqdm(range(NUM_EPISODES)):
        eps = 1.0 / np.sqrt(n + 1)
        total_rewards += [play_one_episode(model, env, eps=eps, gamma=GAMMA, _lambda=LAMBDA)]
        if n % 20 == 0:
            print(f"Episode {n}: total reward: {total_rewards[-1]}, eps: {eps}.")

    # Plot results.
    plot_total_rewards(total_rewards)
    plot_running_average(total_rewards)
    plot_cost_to_go(env, model)

    return 0


if __name__ == "__main__":
    sys.exit(main())
