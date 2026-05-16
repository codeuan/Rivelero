"""
Environment dynamics for the VISTA MDP.

This module simulates the consequences of the agent's actions. It controls
the main MDP loop: reset the problem, receive an action, run the relevant
viewshed calculation, update the state, calculate the reward, and decide
whether the episode has ended.

In simple terms, the policy asks "What should I do next?", while the
environment answers "If you do that, this is what happens."

The environment should coordinate the MDP components, but it should not
contain all of their internal details. State definitions belong in
state.py, action definitions belong in actions.py, reward scoring belongs
in rewards.py, and low-level visibility calculations belong in the
viewshed package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

import numpy as np

from actions import Action, ActionKind, build_actions, get_action


# Type alias for a function that takes a viewpoint and returns a visibility mask.
#
# The returned numpy array should be a 2D boolean array where:
#   True  = this cell is visible from the viewpoint
#   False = this cell is not visible from the viewpoint
ViewshedFunction = Callable[["CandidateViewpoint"], np.ndarray]


@dataclass(frozen=True, slots=True)
class CandidateViewpoint:
    """
    Represents one possible viewpoint the MDP may choose.

    Attributes:
        viewpoint_id:
            Integer ID/index of the viewpoint.

        x:
            X coordinate, longitude, or projected easting.

        y:
            Y coordinate, latitude, or projected northing.

        travel_cost:
            Cost of moving to this viewpoint.

        computation_cost:
            Cost of running the visibility calculation for this viewpoint.
    """

    viewpoint_id: int
    x: float
    y: float
    travel_cost: float = 1.0
    computation_cost: float = 1.0


@dataclass
class MDPState:
    """
    Stores the current state of the MDP.

    Attributes:
        current_viewpoint:
            The currently selected viewpoint.
            None means no viewpoint has been selected yet.

        selected_viewpoints:
            Set of viewpoint IDs that have already been selected.

        visibility_mask:
            Boolean 2D array showing all cells currently covered/visible.

        coverage_percentage:
            Percentage of target cells currently visible.

        redundancy_score:
            Total amount of repeated visibility accumulated so far.

        remaining_budget:
            Remaining movement/computation budget.

        steps_taken:
            Number of actions taken so far.
    """

    current_viewpoint: int | None
    selected_viewpoints: set[int] = field(default_factory=set)
    visibility_mask: np.ndarray | None = None
    coverage_percentage: float = 0.0
    redundancy_score: float = 0.0
    remaining_budget: float = 0.0
    steps_taken: int = 0


class VistaEnvironment:
    """
    Plain-Python environment for the VISTA viewpoint-selection MDP.

    This environment is intentionally not a Gymnasium environment yet.
    It uses a simpler interface while the project logic is still being built.

    Basic usage:

        env = VistaEnvironment(...)
        state = env.reset()

        done = False

        while not done:
            action_number = ...
            state, reward, done, info = env.step(action_number)
    """

    def __init__(
        self,
        candidate_viewpoints: list[CandidateViewpoint],
        viewshed_function: ViewshedFunction,
        map_shape: tuple[int, int],
        *,
        initial_budget: float,
        target_coverage_percentage: float = 95.0,
        max_steps: int | None = None,
        new_area_weight: float = 1.0,
        repeated_area_penalty: float = 0.5,
        travel_cost_weight: float = 1.0,
        computation_cost_weight: float = 1.0,
        target_reached_bonus: float = 10.0,
        stop_early_penalty: float = 10.0,
    ) -> None:
        """
        Create the VISTA MDP environment.

        Args:
            candidate_viewpoints:
                The candidate viewpoints the MDP may choose from.

            viewshed_function:
                Function that calculates the visible cells from a viewpoint.

            map_shape:
                Shape of the visibility map, e.g. (height, width).

            initial_budget:
                Starting budget for travel and computation.

            target_coverage_percentage:
                Coverage percentage at which the episode can be considered
                successful.

            max_steps:
                Optional maximum number of steps before the episode ends.

            new_area_weight:
                Reward weight for newly visible area.

            repeated_area_penalty:
                Penalty weight for repeated visible area.

            travel_cost_weight:
                Penalty weight for movement/travel cost.

            computation_cost_weight:
                Penalty weight for computation cost.

            target_reached_bonus:
                Bonus reward when target coverage is reached.

            stop_early_penalty:
                Penalty applied when the STOP action is chosen before target
                coverage is reached.
        """

        if not candidate_viewpoints:
            raise ValueError("candidate_viewpoints cannot be empty.")

        if initial_budget < 0:
            raise ValueError("initial_budget cannot be negative.")

        self.candidate_viewpoints = candidate_viewpoints
        self.viewshed_function = viewshed_function
        self.map_shape = map_shape

        self.initial_budget = initial_budget
        self.target_coverage_percentage = target_coverage_percentage
        self.max_steps = max_steps

        self.new_area_weight = new_area_weight
        self.repeated_area_penalty = repeated_area_penalty
        self.travel_cost_weight = travel_cost_weight
        self.computation_cost_weight = computation_cost_weight
        self.target_reached_bonus = target_reached_bonus
        self.stop_early_penalty = stop_early_penalty

        self.actions = build_actions(
            candidate_viewpoint_ids=[
                viewpoint.viewpoint_id for viewpoint in candidate_viewpoints
            ],
        )

        self._viewpoints_by_id = {
            viewpoint.viewpoint_id: viewpoint for viewpoint in candidate_viewpoints
        }

        self.state = self.reset()

    def reset(self) -> MDPState:
        """
        Reset the environment to its initial state.

        Returns:
            The initial MDPState.
        """

        self.state = MDPState(
            current_viewpoint=None,
            selected_viewpoints=set(),
            visibility_mask=np.zeros(self.map_shape, dtype=bool),
            coverage_percentage=0.0,
            redundancy_score=0.0,
            remaining_budget=self.initial_budget,
            steps_taken=0,
        )

        return self.state

    def step(self, action_number: int) -> tuple[MDPState, float, bool, dict[str, Any]]:
        """
        Apply one action to the environment.

        Args:
            action_number:
                Integer index of the action chosen by the policy.

        Returns:
            A tuple containing:

                state:
                    The updated MDP state.

                reward:
                    The reward produced by the action.

                done:
                    True if the episode is finished.

                info:
                    Extra debugging information.
        """

        action = get_action(self.actions, action_number)

        if action.kind == ActionKind.STOP:
            return self._handle_stop_action()

        if action.kind == ActionKind.SELECT_VIEWPOINT:
            return self._handle_select_viewpoint_action(action)

        raise ValueError(f"Unsupported action kind: {action.kind}")

    def _handle_stop_action(self) -> tuple[MDPState, float, bool, dict[str, Any]]:
        """
        Handle the STOP action.

        Stopping is always terminal. It is rewarded if target coverage has been
        reached and penalised if the agent stops too early.
        """

        self.state.steps_taken += 1

        target_reached = (
            self.state.coverage_percentage >= self.target_coverage_percentage
        )

        if target_reached:
            reward = self.target_reached_bonus
            reason = "stop action chosen after target coverage reached"
        else:
            reward = -self.stop_early_penalty
            reason = "stop action chosen before target coverage reached"

        done = True

        info = {
            "reason": reason,
            "coverage_percentage": self.state.coverage_percentage,
            "remaining_budget": self.state.remaining_budget,
            "selected_viewpoints": set(self.state.selected_viewpoints),
        }

        return self.state, reward, done, info

    def _handle_select_viewpoint_action(
        self,
        action: Action,
    ) -> tuple[MDPState, float, bool, dict[str, Any]]:
        """
        Handle an action that selects a candidate viewpoint.
        """

        viewpoint_id = action.viewpoint_id

        if viewpoint_id is None:
            raise ValueError("SELECT_VIEWPOINT action is missing viewpoint_id.")

        if viewpoint_id not in self._viewpoints_by_id:
            raise ValueError(f"Unknown viewpoint_id: {viewpoint_id}")

        viewpoint = self._viewpoints_by_id[viewpoint_id]

        if viewpoint_id in self.state.selected_viewpoints:
            reward = -self.stop_early_penalty
            done = self._is_done()

            info = {
                "reason": "viewpoint already selected",
                "viewpoint_id": viewpoint_id,
                "coverage_percentage": self.state.coverage_percentage,
                "remaining_budget": self.state.remaining_budget,
            }

            return self.state, reward, done, info

        action_cost = viewpoint.travel_cost + viewpoint.computation_cost

        if action_cost > self.state.remaining_budget:
            reward = -action_cost
            done = True

            info = {
                "reason": "budget exhausted or action too expensive",
                "viewpoint_id": viewpoint_id,
                "action_cost": action_cost,
                "remaining_budget": self.state.remaining_budget,
            }

            return self.state, reward, done, info

        previous_visibility = self.state.visibility_mask

        if previous_visibility is None:
            raise RuntimeError("State visibility_mask has not been initialised.")

        new_visibility = self.viewshed_function(viewpoint)

        self._validate_visibility_mask(new_visibility)

        newly_visible_mask = np.logical_and(new_visibility, ~previous_visibility)
        repeated_visible_mask = np.logical_and(new_visibility, previous_visibility)

        newly_visible_area = int(np.count_nonzero(newly_visible_mask))
        repeated_visible_area = int(np.count_nonzero(repeated_visible_mask))

        updated_visibility = np.logical_or(previous_visibility, new_visibility)

        self.state.current_viewpoint = viewpoint_id
        self.state.selected_viewpoints.add(viewpoint_id)
        self.state.visibility_mask = updated_visibility
        self.state.coverage_percentage = self._calculate_coverage_percentage(
            updated_visibility
        )
        self.state.redundancy_score += repeated_visible_area
        self.state.remaining_budget -= action_cost
        self.state.steps_taken += 1

        reward = self._calculate_select_viewpoint_reward(
            newly_visible_area=newly_visible_area,
            repeated_visible_area=repeated_visible_area,
            travel_cost=viewpoint.travel_cost,
            computation_cost=viewpoint.computation_cost,
        )

        target_reached = (
            self.state.coverage_percentage >= self.target_coverage_percentage
        )

        if target_reached:
            reward += self.target_reached_bonus

        done = self._is_done()

        info = {
            "reason": self._get_done_reason() if done else "step completed",
            "viewpoint_id": viewpoint_id,
            "newly_visible_area": newly_visible_area,
            "repeated_visible_area": repeated_visible_area,
            "coverage_percentage": self.state.coverage_percentage,
            "redundancy_score": self.state.redundancy_score,
            "remaining_budget": self.state.remaining_budget,
            "selected_viewpoints": set(self.state.selected_viewpoints),
        }

        return self.state, reward, done, info

    def _calculate_select_viewpoint_reward(
        self,
        *,
        newly_visible_area: int,
        repeated_visible_area: int,
        travel_cost: float,
        computation_cost: float,
    ) -> float:
        """
        Calculate reward for selecting a viewpoint.

        Reward structure:

            + new visible area
            - repeated visible area
            - distance/travel cost
            - computation cost
        """

        reward = 0.0

        reward += self.new_area_weight * newly_visible_area
        reward -= self.repeated_area_penalty * repeated_visible_area
        reward -= self.travel_cost_weight * travel_cost
        reward -= self.computation_cost_weight * computation_cost

        return reward

    def _calculate_coverage_percentage(self, visibility_mask: np.ndarray) -> float:
        """
        Calculate the percentage of map cells currently visible.
        """

        total_cells = visibility_mask.size

        if total_cells == 0:
            return 0.0

        visible_cells = int(np.count_nonzero(visibility_mask))

        return (visible_cells / total_cells) * 100.0

    def _validate_visibility_mask(self, visibility_mask: np.ndarray) -> None:
        """
        Check that a viewshed result has the correct shape and type.
        """

        if not isinstance(visibility_mask, np.ndarray):
            raise TypeError("viewshed_function must return a numpy array.")

        if visibility_mask.shape != self.map_shape:
            raise ValueError(
                f"viewshed_function returned shape {visibility_mask.shape}, "
                f"but expected {self.map_shape}."
            )

        if visibility_mask.dtype != bool:
            raise TypeError("viewshed_function must return a boolean numpy array.")

    def _is_done(self) -> bool:
        """
        Return True if the episode should end.
        """

        if self.state.coverage_percentage >= self.target_coverage_percentage:
            return True

        if self.state.remaining_budget <= 0:
            return True

        if self.max_steps is not None and self.state.steps_taken >= self.max_steps:
            return True

        return False

    def _get_done_reason(self) -> str:
        """
        Return a readable reason for why the episode has ended.
        """

        if self.state.coverage_percentage >= self.target_coverage_percentage:
            return "target coverage reached"

        if self.state.remaining_budget <= 0:
            return "budget exhausted"

        if self.max_steps is not None and self.state.steps_taken >= self.max_steps:
            return "maximum steps reached"

        return "not done"

    def get_valid_action_numbers(self) -> list[int]:
        """
        Return the action numbers that are currently valid.

        STOP is always valid.

        A SELECT_VIEWPOINT action is valid if:
            - the viewpoint has not already been selected
            - the viewpoint cost does not exceed the remaining budget
        """

        valid_action_numbers: list[int] = []

        for action_number, action in enumerate(self.actions):
            if action.kind == ActionKind.STOP:
                valid_action_numbers.append(action_number)
                continue

            if action.viewpoint_id is None:
                continue

            if action.viewpoint_id in self.state.selected_viewpoints:
                continue

            viewpoint = self._viewpoints_by_id[action.viewpoint_id]
            action_cost = viewpoint.travel_cost + viewpoint.computation_cost

            if action_cost <= self.state.remaining_budget:
                valid_action_numbers.append(action_number)

        return valid_action_numbers

    def get_observation(self) -> dict[str, Any]:
        """
        Return a simple dictionary version of the current state.

        This is useful for policies that do not need direct access to the full
        MDPState object.
        """

        return {
            "current_viewpoint": self.state.current_viewpoint,
            "selected_viewpoints": set(self.state.selected_viewpoints),
            "coverage_percentage": self.state.coverage_percentage,
            "redundancy_score": self.state.redundancy_score,
            "remaining_budget": self.state.remaining_budget,
            "steps_taken": self.state.steps_taken,
        }