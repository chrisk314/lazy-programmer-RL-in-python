from typing import Dict, List, Tuple


IntVec2d = Tuple[int, int]
RewardsDict = Dict[IntVec2d, float]
ActionsDict = Dict[IntVec2d, List[IntVec2d]]

REWARDS: RewardsDict = {
    (0, 3): 1,
    (1, 3): -1,
}
_U, _R, _D, _L = (-1, 0), (0, 1), (1, 0), (0, -1)
ACTIONS: ActionsDict = {
    (0, 0): [_R, _D],
    (0, 1): [_R, _L],
    (0, 2): [_R, _D, _L],
    (1, 0): [_U, _D],
    (1, 2): [_U, _R, _D],
    (2, 0): [_U, _R],
    (2, 1): [_R, _L],
    (2, 2): [_U, _R, _L],
    (2, 3): [_U, _L],
}


class GridWorld:
    """`GridWorld` represents a 2D grid environment for an RL agent."""

    def __init__(
        self, rows: int, cols: int, start: IntVec2d, actions: ActionsDict, rewards: RewardsDict
    ) -> None:
        """Instantiates `GridWorld`.

        Args:
            rows: Number of rows in the grid (y axis).
            cols: Number of columns in the grid (x axis).
            start: Starting location in the grid.
        """
        self.rows = rows
        self.cols = cols
        self.state = start
        self.actions = actions
        self.rewards = rewards
        self._active_states = set(self.actions.keys()) | set(self.rewards.keys())

    def _update_state(self, action: IntVec2d) -> IntVec2d:
        i, j = self.state
        di, dj = action
        new_state = (i + di, j + dj)
        in_x = 0 <= new_state[0] < self.cols
        in_y = 0 <= new_state[1] < self.rows
        if in_x and in_y:
            self.state = new_state
            return new_state
        raise ValueError("Action takes state out of bounds.")

    def act(self, action: IntVec2d) -> Tuple[IntVec2d, float]:
        """Takes specified `action` and returns new state and reward."""
        if action not in self.actions:
            return self.state, 0.0
        self.state = self._update_state(action)
        reward = self.rewards.get(self.state)
        if reward is None:
            raise ValueError("Reached disallowed state!")
        return self.state, reward

    def is_active(self) -> bool:
        """Returns True iff the episode is active."""
        return self.state in self._active_states


def policy(state: IntVec2d) -> IntVec2d:
    """Returns action to take based on state."""
    pass


def main() -> None:
    """Entrypoint for RL run."""
    E = GridWorld(3, 4, (0, 0), ACTIONS, REWARDS)
    s = E.state
    while E.is_active():
        a = policy(s)
        s, r = E.act(a)


if __name__ == "__main__":
    main()
