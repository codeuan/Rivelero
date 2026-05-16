"""
State definitions for the VISTA MDP.

This module defines what the agent knows at a given decision point. A state
should contain the information needed to choose the next action, such as
the current viewpoint, the current coverage map, the remaining budget, and
which viewpoints have already been selected.

The state should summarise the relevant past. It should not store every
temporary calculation unless that information is needed for future decisions.
"""


from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class MDPState:
    """
    Stores the current state of the VISTA MDP.

    Attributes:
        current_viewpoint:
            The ID of the currently selected viewpoint.
            None means no viewpoint has been selected yet.

        selected_viewpoints:
            Set of viewpoint IDs that have already been selected.

        visibility_mask:
            Boolean 2D array showing which map cells are currently visible.

            True  = visible
            False = not visible

        coverage_percentage:
            Percentage of the map/target area currently visible.

        redundancy_score:
            Total amount of repeated visibility accumulated so far.

            For example, if a new viewpoint sees cells that were already seen
            by previous viewpoints, those repeated cells contribute to
            redundancy.

        remaining_budget:
            Remaining movement/computation budget.

        steps_taken:
            Number of actions taken so far.
    """

    current_viewpoint: int | None = None
    selected_viewpoints: set[int] = field(default_factory=set)
    visibility_mask: np.ndarray | None = None
    coverage_percentage: float = 0.0
    redundancy_score: float = 0.0
    remaining_budget: float = 0.0
    steps_taken: int = 0

    @classmethod
    def create_initial_state(
        cls,
        *,
        map_shape: tuple[int, int],
        initial_budget: float,
    ) -> "MDPState":
        """
        Create the initial MDP state.

        Args:
            map_shape:
                Shape of the visibility map, e.g. (height, width).

            initial_budget:
                Starting movement/computation budget.

        Returns:
            A fresh MDPState object.
        """

        if initial_budget < 0:
            raise ValueError("initial_budget cannot be negative.")

        if len(map_shape) != 2:
            raise ValueError("map_shape must be a 2D shape, e.g. (height, width).")

        height, width = map_shape

        if height <= 0 or width <= 0:
            raise ValueError("map_shape dimensions must be positive.")

        return cls(
            current_viewpoint=None,
            selected_viewpoints=set(),
            visibility_mask=np.zeros(map_shape, dtype=bool),
            coverage_percentage=0.0,
            redundancy_score=0.0,
            remaining_budget=initial_budget,
            steps_taken=0,
        )

    def copy(self) -> "MDPState":
        """
        Return a safe copy of the state.

        This is useful if a policy wants to simulate something without changing
        the real environment state.
        """

        return MDPState(
            current_viewpoint=self.current_viewpoint,
            selected_viewpoints=set(self.selected_viewpoints),
            visibility_mask=(
                None if self.visibility_mask is None else self.visibility_mask.copy()
            ),
            coverage_percentage=self.coverage_percentage,
            redundancy_score=self.redundancy_score,
            remaining_budget=self.remaining_budget,
            steps_taken=self.steps_taken,
        )

    def has_selected_viewpoint(self, viewpoint_id: int) -> bool:
        """
        Return True if a viewpoint has already been selected.
        """

        return viewpoint_id in self.selected_viewpoints

    def mark_viewpoint_selected(self, viewpoint_id: int) -> None:
        """
        Mark a viewpoint as selected and make it the current viewpoint.
        """

        self.current_viewpoint = viewpoint_id
        self.selected_viewpoints.add(viewpoint_id)

    def reduce_budget(self, amount: float) -> None:
        """
        Reduce the remaining budget.

        Args:
            amount:
                Amount to subtract from the budget.
        """

        if amount < 0:
            raise ValueError("Budget reduction amount cannot be negative.")

        self.remaining_budget -= amount

    def increment_steps(self) -> None:
        """
        Increase the step counter by one.
        """

        self.steps_taken += 1

    def update_visibility(
        self,
        *,
        new_visibility_mask: np.ndarray,
    ) -> tuple[int, int]:
        """
        Update the state's visibility mask.

        Args:
            new_visibility_mask:
                Boolean 2D array produced by a viewshed calculation.

        Returns:
            A tuple:

                newly_visible_area:
                    Number of cells newly seen for the first time.

                repeated_visible_area:
                    Number of cells that were already visible before this
                    update.
        """

        if self.visibility_mask is None:
            raise RuntimeError("visibility_mask has not been initialised.")

        if not isinstance(new_visibility_mask, np.ndarray):
            raise TypeError("new_visibility_mask must be a numpy array.")

        if new_visibility_mask.dtype != bool:
            raise TypeError("new_visibility_mask must be a boolean array.")

        if new_visibility_mask.shape != self.visibility_mask.shape:
            raise ValueError(
                f"new_visibility_mask has shape {new_visibility_mask.shape}, "
                f"but expected {self.visibility_mask.shape}."
            )

        newly_visible_mask = np.logical_and(
            new_visibility_mask,
            ~self.visibility_mask,
        )

        repeated_visible_mask = np.logical_and(
            new_visibility_mask,
            self.visibility_mask,
        )

        newly_visible_area = int(np.count_nonzero(newly_visible_mask))
        repeated_visible_area = int(np.count_nonzero(repeated_visible_mask))

        self.visibility_mask = np.logical_or(
            self.visibility_mask,
            new_visibility_mask,
        )

        self.coverage_percentage = self.calculate_coverage_percentage()
        self.redundancy_score += repeated_visible_area

        return newly_visible_area, repeated_visible_area

    def calculate_coverage_percentage(self) -> float:
        """
        Calculate the percentage of currently visible cells.
        """

        if self.visibility_mask is None:
            return 0.0

        total_cells = self.visibility_mask.size

        if total_cells == 0:
            return 0.0

        visible_cells = int(np.count_nonzero(self.visibility_mask))

        return (visible_cells / total_cells) * 100.0

    def target_coverage_reached(self, target_coverage_percentage: float) -> bool:
        """
        Return True if the target coverage percentage has been reached.
        """

        return self.coverage_percentage >= target_coverage_percentage

    def budget_exhausted(self) -> bool:
        """
        Return True if no budget remains.
        """

        return self.remaining_budget <= 0

    def as_observation(self) -> dict[str, Any]:
        """
        Return a simple dictionary representation of the state.

        This is useful for policies, logging, or debugging.
        """

        return {
            "current_viewpoint": self.current_viewpoint,
            "selected_viewpoints": set(self.selected_viewpoints),
            "coverage_percentage": self.coverage_percentage,
            "redundancy_score": self.redundancy_score,
            "remaining_budget": self.remaining_budget,
            "steps_taken": self.steps_taken,
        }

    def as_numeric_observation(self) -> np.ndarray:
        """
        Return a small numeric observation vector.

        This may be useful later if you convert the environment to Gymnasium
        or use a machine-learning policy.

        Format:
            [
                current_viewpoint,
                coverage_percentage,
                redundancy_score,
                remaining_budget,
                steps_taken,
            ]

        If no viewpoint has been selected yet, current_viewpoint is represented
        as -1.
        """

        current_viewpoint_value = (
            -1 if self.current_viewpoint is None else self.current_viewpoint
        )

        return np.array(
            [
                current_viewpoint_value,
                self.coverage_percentage,
                self.redundancy_score,
                self.remaining_budget,
                self.steps_taken,
            ],
            dtype=np.float32,
        )