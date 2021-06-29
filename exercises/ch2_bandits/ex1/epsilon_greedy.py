#!/usr/bin/env python3
"""This script explores the epsilon greedy algorithm for controlling
the explore vs exploit decision process.
"""
from collections import deque
from pprint import pprint
import string
from typing import Deque, List, Tuple

from matplotlib import pyplot as plt
import nptyping as npt
import numpy as np
from numpy import random as npr


CHAR_CHOICE = list(string.ascii_lowercase + string.digits)
EPSILON = 0.05
NUM_TRIALS = 10000
SEED = 0


def rand_str(size: int = 4) -> str:
    """Returns a random string of fixed `size`."""
    return "".join([npr.choice(CHAR_CHOICE) for _ in range(size)])


class Bandit(object):
    """`Bandit` represents a \"one-armed bandit\" slot machine."""

    def __init__(self, p: float = 1.0) -> None:
        """Instantiates `Bandit`."""
        self.p = p
        self._id = rand_str()

    def pull(self) -> float:
        """Returns a random outcome for the `Bandit`."""
        return float(int(npr.random() < self.p))

    def __str__(self) -> str:
        return f"{self.__class__.__name__}-{self._id}"


class MLE(object):
    """`MLE` stores and updates Maximum Likelihood Estimate."""

    def __init__(self, est_init: float = 0.0, trials_init: int = 0) -> None:
        self.est = est_init
        self.trials = trials_init

    def update(self, obs: float) -> float:
        """Updates MLE with an observation and returns new value."""
        self.est = (self.trials * self.est + obs) / (self.trials + 1)
        self.trials += 1
        return self.est

    def __str__(self) -> str:
        return f"{repr(self)}: est: {self.est:1.3e}, trials: {self.trials}"


def update_epsilon(eps_0: float, iter: int) -> float:
    """Updates epsilon based on inital value and iteration count."""
    alpha = 0.99
    return eps_0 * alpha ** iter


def run_experiment(
    bandits: List[Bandit], eps: float, num_trials: int
) -> Tuple[npt.NDArray[(NUM_TRIALS,), npt.Float64], List[MLE]]:
    """Runs an experiment on a set of bandits.

    Args:
        bandits: Bandits to extract rewards from.
        eps: Probability of exploration vs. exploitation.
        num_trials: Number of observations to make of the bandits.

    Returns:
        Tuple containing the reward vector for all trials and a
            list of maximum likelihood estimates for the bandits.
    """
    # Initialise `Bandit` MLE and reward state
    bandit_mle: List[MLE] = [MLE() for _ in bandits]
    rewards: npt.NDArray = np.empty(num_trials)

    n_bandits = len(bandits)

    eps_0 = eps  # Initial value of epsilon

    # Perform a number of trials on the set of `Bandits`.
    for i in range(num_trials):
        # Draw random number. Choose between explore vs exploit.
        # eps = update_epsilon(eps_0, i)
        if npr.random() < eps:
            b_idx = npr.choice(n_bandits)
        else:
            b_idx = np.argmax([x.est for x in bandit_mle])
        b = bandits[b_idx]
        # Make an observation
        obs = b.pull()
        rewards[i] = obs
        bandit_mle[b_idx].update(obs)

    return rewards, bandit_mle


def plot_rewards(rewards: List[float], eps: float) -> None:
    """Plot rewards for current epsilon."""
    cum_rewards = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    plt.plot(cum_rewards, label=f"{eps=}")


def display_plot() -> None:
    """Helper method to configure plot and display."""
    plt.xlabel("iters")
    plt.ylabel("average reward")
    plt.xscale("log")
    plt.legend()
    plt.grid(which="both")
    plt.show()


if __name__ == "__main__":
    # Create a set of `Bandits`.
    bandit_config = [0.1, 0.5, 0.8]
    bandits = [Bandit(p=p) for p in bandit_config]

    # Print some data about the bandits.
    print("Bandit, p")
    for b in bandits:
        print(b, b.p)

    # Perform several experiments with different values of epsilon
    for eps in (0.01, 0.05, 0.10):
        print(f"\nepsilon={eps} ---------------")
        rewards, bandit_mle = run_experiment(bandits, eps, NUM_TRIALS)
        avg_reward = sum(rewards) / NUM_TRIALS
        print(f"Avg reward: {avg_reward}")

        pprint([str(mle) for mle in bandit_mle])

        b_idx = np.argmax([x.est for x in bandit_mle])
        best_bandit = bandits[b_idx]
        print(f"Best bandit: {best_bandit}")

        # TODO : Add plotting.
        plot_rewards(rewards, eps)

    display_plot()
