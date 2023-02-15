"""This script trains and runs an RL agent to solve the mountain car control problem.

The state and action space is treated as continuous split into bins. Several RBFSamplers
are stacked together to produce features from the input data.

The N-step method is used to evaluate the return, G, and update the model.
"""

from collections import deque
import sys

from gym import Env, make
import numpy as np
from sklearn.linear_model import SGDRegressor
from tqdm import tqdm
from ..ch3_basic_RL.mountain_car_rbf import (
    ALPHA,
    EPSILON,
    FeatureTransformer,
    GAMMA,
    MAX_ITERS,
    Model,
    NUM_EPISODES,
    plot_cost_to_go,
    plot_running_average,
    plot_total_rewards,
)


class SGDRegressor:
    def __init__(self, learning_rate: float = ALPHA) -> None:
        self.w = None
        self.learning_rate = learning_rate

    def partial_fit(self, X, Y) -> None:
        if self.w is None:
            self.w = np.random.randn(X.shape[1]) / np.sqrt(np.product(X.shape[1]))
        self.w += self.learning_rate * np.dot(Y - np.dot(X, self.w), X)

    def predict(self, X) -> np.ndarray:
        return np.dot(X, self.w)


def play_one_episode(
    model: Model, env: Env, eps: float = EPSILON, gamma: float = GAMMA, N: int = 5
) -> float:
    _s, _ = env.reset()
    done = False
    iters = 0
    total_reward = 0.0

    s = deque(maxlen=N)
    a = deque(maxlen=N)
    r = deque(maxlen=N)

    _gamma_vec = [gamma**i for i in range(N + 1)]

    while not done and iters < MAX_ITERS:
        # Choose action and take a step
        _a = model.sample_action(_s, eps=eps)
        s2, _r, done, _, _ = env.step(_a)

        s += [_s]
        a += [_a]
        r += [_r]

        if len(s) == N:
            # Update the model with N-steps of data
            G = np.dot(_gamma_vec, np.array([*r, max(model.predict(s2))]))
            model.update(s[0], a[0], G)

        # Update iteration vars
        _s = s2
        total_reward += _r
        iters += 1

    # Episodes terminate after 200 steps. We reached the goal if position > 0.5
    # for the final state.
    if _s[0] > 0.5:
        while len(s) > 0:
            G = np.dot(_gamma_vec[: len(s)], r)
            model.update(s[0], a[0], G)
            s.popleft()
            a.popleft()
            r.popleft()
    else:
        # Never reached the end so guess the future rewards as never succeeding.
        while len(s) > 0:
            r_guess = r + [-1] * (N - len(r))
            G = np.dot(_gamma_vec[: len(s)], r_guess)
            model.update(s[0], a[0], G)
            s.popleft()
            a.popleft()
            r.popleft()

    return total_reward


def main() -> int:
    # Set up environment and agent.
    env = make("MountainCar-v0")
    trans = FeatureTransformer(env)
    model = Model(env, trans, learning_rate=ALPHA, regressor_cls=SGDRegressor)
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
