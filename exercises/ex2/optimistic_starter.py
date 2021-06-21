#!/usr/bin/env python3
"""This script explores the optimistic initial value algorithm for controlling
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
from numpy.lib.twodim_base import tri


CHAR_CHOICE = list(string.ascii_lowercase + string.digits)
NUM_TRIALS = 10000
SEED = 0


def rand_str(size: int = 4) -> str:
    """Returns a random string of fixed `size`."""
    return "".join([npr.choice(CHAR_CHOICE) for _ in range(size)])


class Bandit(object):
    """`Bandit` represents a \"one-armed bandit\" slot machine."""

    def __init__(self, loc: float = 1.0, scale: float = 0.5) -> None:
        """Instantiates `Bandit`."""
        self.loc = loc
        self.scale = scale
        self._id = rand_str()

    def pull(self) -> float:
        """Returns a random outcome for the `Bandit`."""
        return npr.normal(loc=self.loc, scale=self.scale)

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


def run_experiment(
    bandits: List[Bandit], opt_est: float, num_trials: int
) -> Tuple[npt.NDArray[(NUM_TRIALS,), npt.Float64], List[MLE]]:
    """Runs an experiment on a set of bandits.

    Args:
        bandits: Bandits to extract rewards from.
        opt_est: Initial optimistic estimate for MLEs.
        num_trials: Number of observations to make of the bandits.

    Returns:
        Tuple containing the reward vector for all trials and a
            list of maximum likelihood estimates for the bandits.
    """
    # Initialise `Bandit` MLE and reward state
    bandit_mle: List[MLE] = [MLE(est_init=opt_est, trials_init=1) for _ in bandits]
    rewards: npt.NDArray = np.empty(num_trials)

    # Perform a number of trials on the set of `Bandits`.
    for i in range(num_trials):
        # Draw random number. Choose between explore vs exploit.
        b_idx = np.argmax([x.est for x in bandit_mle])
        b = bandits[b_idx]
        # Make an observation
        obs = b.pull()
        rewards[i] = obs
        bandit_mle[b_idx].update(obs)

    return rewards, bandit_mle


def plot_rewards(rewards: List[float], opt_est: float) -> None:
    """Plot rewards for current optimistic estimate."""
    cum_rewards = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
    plt.plot(cum_rewards, label=f"{opt_est=}")


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
    bandit_config = [(0.1, 0.5), (0.8, 0.2), (0.5, 1.0), (0.6, 0.3), (0.9, 0.1)]
    bandits = [Bandit(loc=loc, scale=scale) for loc, scale in bandit_config]

    # Print some data about the bandits.
    print("Bandit, mean, stddev")
    for b in bandits:
        print(b, b.loc, b.scale)

    # Perform several experiments with different values of epsilon
    for opt_est in (1, 10, 100):
        print(f"\noptimistic estimate: {opt_est} ---------------")
        rewards, bandit_mle = run_experiment(bandits, opt_est, NUM_TRIALS)
        avg_reward = sum(rewards) / NUM_TRIALS
        print(f"Avg reward: {avg_reward}")

        pprint([str(mle) for mle in bandit_mle])

        b_idx = np.argmax([x.est for x in bandit_mle])
        best_bandit = bandits[b_idx]
        print(f"Best bandit: {best_bandit}")

        # TODO : Add plotting.
        plot_rewards(rewards, opt_est)

    display_plot()
