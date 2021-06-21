#!/usr/bin/env python3
from collections import deque
from pprint import pprint
import string
from typing import Deque, List, Tuple

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
    bandits: List[Bandit], eps: float, num_trials: int
) -> Tuple[List[float], List[MLE]]:

    # Print some data about the bandits.
    print("Bandit, mean, stddev")
    for b in bandits:
        print(b, b.loc, b.scale)

    # Initialise `Bandit` MLE and reward state
    bandit_mle = [MLE() for _ in bandits]
    rewards: Deque = deque()

    n_bandits = len(bandits)

    # Perform a number of trials on the set of `Bandits`.
    for _ in range(num_trials):
        # Draw random number. Choose between explore vs exploit.
        if npr.random() < eps:
            b_idx = npr.choice(n_bandits)
        else:
            b_idx = np.argmax([x.est for x in bandit_mle])
        b = bandits[b_idx]
        # Make an observation
        obs = b.pull()
        rewards.append(obs)
        bandit_mle[b_idx].update(obs)

    return list(rewards), bandit_mle


if __name__ == "__main__":
    # Create a set of `Bandits`.
    bandit_config = [(1.0, 0.5), (0.8, 0.2), (0.5, 1.0)]
    bandits = [Bandit(loc=loc, scale=scale) for loc, scale in bandit_config]

    rewards, bandit_mle = run_experiment(bandits, EPSILON, NUM_TRIALS)
    avg_reward = sum(rewards) / NUM_TRIALS
    print(f"Avg reward: {avg_reward}")

    pprint([str(mle) for mle in bandit_mle])

    b_idx = np.argmax([x.est for x in bandit_mle])
    best_bandit = bandits[b_idx]
    print(f"Best bandit: {best_bandit}")
