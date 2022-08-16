from typing import Dict, Set, Tuple


IntVec2d = Tuple[int, int]
RewardsDict = Dict[IntVec2d, float]
ActionsDict = Dict[IntVec2d, Set[IntVec2d]]

REWARDS: RewardsDict = {
    (0, 3): 1,
    (1, 3): -1,
}
_U, _R, _D, _L = (-1, 0), (0, 1), (1, 0), (0, -1)
ACTIONS: ActionsDict = {
    (0, 0): set([_R, _D]),
    (0, 1): set([_R, _L]),
    (0, 2): set([_R, _D, _L]),
    (1, 0): set([_U, _D]),
    (1, 2): set([_U, _R, _D]),
    (2, 0): set([_U, _R]),
    (2, 1): set([_R, _L]),
    (2, 2): set([_U, _R, _L]),
    (2, 3): set([_U, _L]),
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
            actions: Possible actions for the agent.
            rewards: Rewards for the states.
        """
        self._rows: int = rows
        self._cols: int = cols
        self._state: IntVec2d = start
        self._actions: ActionsDict = actions
        self._rewards: RewardsDict = rewards

    @property
    def state(self) -> IntVec2d:
        return self._state

    def _update_state(self, action: IntVec2d) -> IntVec2d:
        i, j = self._state
        di, dj = action
        new__state = (i + di, j + dj)
        in_x = 0 <= new__state[0] < self._cols
        in_y = 0 <= new__state[1] < self._rows
        if in_x and in_y:
            self._state = new__state
            return new__state
        raise ValueError("Action takes _state out of bounds.")

    def act(self, action: IntVec2d) -> Tuple[IntVec2d, float]:
        """Takes specified `action` and returns new _state and reward."""
        if action not in self._actions[self._state]:
            return self._state, 0.0
        self._state = self._update_state(action)
        reward = self._rewards.get(self._state, 0.0)
        return self._state, reward

    def is_active(self) -> bool:
        """Returns True iff the episode is active."""
        return self._state in self._actions.keys()


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
