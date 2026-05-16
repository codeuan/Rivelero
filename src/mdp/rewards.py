"""
Reward functions for the VISTA MDP.

This module defines how the agent is scored after taking an action. The
reward function translates VISTA's goals into numbers: reward new useful
visibility, penalise redundant coverage, penalise travel or computation
costs, and reward stopping once enough coverage has been achieved.

Reward logic is kept separate because it is likely to be adjusted often
during experimentation.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RewardWeights:
    """
    Stores the weighting values used by the reward function.

    Attributes:
        new_area_weight:
            Reward per newly visible cell.

        repeated_area_penalty:
            Penalty per already-visible cell seen again.

        travel_cost_weight:
            Penalty multiplier for travel/movement cost.

        computation_cost_weight:
            Penalty multiplier for computational cost.

        target_reached_bonus:
            Bonus added when the target coverage percentage is reached.

        stop_success_bonus:
            Reward for choosing STOP after the target has already been reached.

        stop_early_penalty:
            Penalty for choosing STOP before the target has been reached.

        invalid_action_penalty:
            Penalty for choosing an invalid action, such as revisiting a
            viewpoint.

        budget_failure_penalty:
            Extra penalty for choosing an action that exceeds the remaining
            budget.
    """

    new_area_weight: float = 1.0
    repeated_area_penalty: float = 0.5
    travel_cost_weight: float = 1.0
    computation_cost_weight: float = 1.0
    target_reached_bonus: float = 10.0
    stop_success_bonus: float = 10.0
    stop_early_penalty: float = 10.0
    invalid_action_penalty: float = 10.0
    budget_failure_penalty: float = 10.0


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """
    Stores the components of a calculated reward.

    This is useful for debugging because it lets you see why the MDP received
    a particular score.
    """

    new_area_reward: float = 0.0
    repeated_area_penalty: float = 0.0
    travel_cost_penalty: float = 0.0
    computation_cost_penalty: float = 0.0
    target_reached_bonus: float = 0.0
    stop_reward: float = 0.0
    invalid_action_penalty: float = 0.0
    budget_failure_penalty: float = 0.0

    @property
    def total(self) -> float:
        """
        Return the final reward total.
        """

        return (
            self.new_area_reward
            - self.repeated_area_penalty
            - self.travel_cost_penalty
            - self.computation_cost_penalty
            + self.target_reached_bonus
            + self.stop_reward
            - self.invalid_action_penalty
            - self.budget_failure_penalty
        )

    def as_dict(self) -> dict[str, float]:
        """
        Return the reward breakdown as a dictionary.

        Useful for adding to the environment info dictionary.
        """

        return {
            "new_area_reward": self.new_area_reward,
            "repeated_area_penalty": self.repeated_area_penalty,
            "travel_cost_penalty": self.travel_cost_penalty,
            "computation_cost_penalty": self.computation_cost_penalty,
            "target_reached_bonus": self.target_reached_bonus,
            "stop_reward": self.stop_reward,
            "invalid_action_penalty": self.invalid_action_penalty,
            "budget_failure_penalty": self.budget_failure_penalty,
            "total_reward": self.total,
        }


def calculate_select_viewpoint_reward(
    *,
    newly_visible_area: int,
    repeated_visible_area: int,
    travel_cost: float,
    computation_cost: float,
    target_reached: bool,
    weights: RewardWeights,
) -> RewardBreakdown:
    """
    Calculate the reward for selecting a viewpoint.

    The basic formula is:

        reward =
            + newly visible area
            - repeated visible area
            - travel cost
            - computation cost
            + target reached bonus, if relevant

    Args:
        newly_visible_area:
            Number of map cells that became visible for the first time.

        repeated_visible_area:
            Number of map cells that were visible already and were seen again.

        travel_cost:
            Cost of moving to the selected viewpoint.

        computation_cost:
            Cost of running the viewshed/visibility calculation.

        target_reached:
            Whether the action caused the target coverage level to be reached.

        weights:
            RewardWeights object controlling reward/penalty strength.

    Returns:
        A RewardBreakdown object.
    """

    if newly_visible_area < 0:
        raise ValueError("newly_visible_area cannot be negative.")

    if repeated_visible_area < 0:
        raise ValueError("repeated_visible_area cannot be negative.")

    if travel_cost < 0:
        raise ValueError("travel_cost cannot be negative.")

    if computation_cost < 0:
        raise ValueError("computation_cost cannot be negative.")

    return RewardBreakdown(
        new_area_reward=weights.new_area_weight * newly_visible_area,
        repeated_area_penalty=weights.repeated_area_penalty * repeated_visible_area,
        travel_cost_penalty=weights.travel_cost_weight * travel_cost,
        computation_cost_penalty=weights.computation_cost_weight * computation_cost,
        target_reached_bonus=(
            weights.target_reached_bonus if target_reached else 0.0
        ),
    )


def calculate_stop_reward(
    *,
    target_reached: bool,
    weights: RewardWeights,
) -> RewardBreakdown:
    """
    Calculate the reward for choosing the STOP action.

    Stopping is good if the target has already been reached.
    Stopping is bad if the target has not been reached.
    """

    if target_reached:
        return RewardBreakdown(stop_reward=weights.stop_success_bonus)

    return RewardBreakdown(
        invalid_action_penalty=weights.stop_early_penalty,
    )


def calculate_invalid_action_reward(
    *,
    weights: RewardWeights,
) -> RewardBreakdown:
    """
    Calculate the penalty for an invalid action.

    Example invalid actions:
        - selecting a viewpoint that has already been selected
        - selecting a viewpoint that does not exist
    """

    return RewardBreakdown(
        invalid_action_penalty=weights.invalid_action_penalty,
    )


def calculate_budget_failure_reward(
    *,
    action_cost: float,
    weights: RewardWeights,
) -> RewardBreakdown:
    """
    Calculate the penalty when an action exceeds the remaining budget.

    The penalty includes:
        - a fixed budget failure penalty
        - the attempted action cost
    """

    if action_cost < 0:
        raise ValueError("action_cost cannot be negative.")

    return RewardBreakdown(
        budget_failure_penalty=weights.budget_failure_penalty + action_cost,
    )


def calculate_coverage_percentage(
    *,
    visible_cells: int,
    total_cells: int,
) -> float:
    """
    Calculate coverage percentage.

    Args:
        visible_cells:
            Number of currently visible cells.

        total_cells:
            Total number of cells in the target map.

    Returns:
        Coverage as a percentage from 0 to 100.
    """

    if visible_cells < 0:
        raise ValueError("visible_cells cannot be negative.")

    if total_cells <= 0:
        raise ValueError("total_cells must be greater than 0.")

    if visible_cells > total_cells:
        raise ValueError("visible_cells cannot be greater than total_cells.")

    return (visible_cells / total_cells) * 100.0


def has_reached_target_coverage(
    *,
    coverage_percentage: float,
    target_coverage_percentage: float,
) -> bool:
    """
    Return True if the target coverage percentage has been reached.
    """

    return coverage_percentage >= target_coverage_percentage