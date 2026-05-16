"""
Policy definitions for the VISTA MDP.

This module defines how the agent chooses actions. A policy receives the
current state/environment and selects the next action, such as choosing the
candidate viewpoint with the best expected value or deciding to stop.

"""


from __future__ import annotations

import random
from typing import Protocol, Any

import numpy as np

from actions import ActionKind, get_action, describe_action


class Policy(Protocol):
    """
    Interface for MDP policies.

    Any policy class should provide a choose_action method that receives the
    current environment and returns an integer action number.
    """

    def choose_action(self, env: Any) -> int:
        """
        Choose the next action number.
        """
        ...


class RandomPolicy:
    """
    Choose randomly from the currently valid actions.

    This is useful for testing whether the MDP loop works before adding a
    clever strategy.
    """

    def __init__(self, *, seed: int | None = None, allow_stop: bool = True) -> None:
        self.random_generator = random.Random(seed)
        self.allow_stop = allow_stop

    def choose_action(self, env: Any) -> int:
        """
        Choose a random valid action number.
        """

        valid_action_numbers = env.get_valid_action_numbers()

        if not self.allow_stop:
            valid_action_numbers = [
                action_number
                for action_number in valid_action_numbers
                if get_action(env.actions, action_number).kind != ActionKind.STOP
            ]

        if not valid_action_numbers:
            raise RuntimeError("No valid actions are available.")

        return self.random_generator.choice(valid_action_numbers)


class SequentialPolicy:
    """
    Choose the first valid viewpoint action.

    This is not intelligent. It is mainly useful for debugging because it gives
    predictable behaviour.
    """

    def __init__(self, *, allow_stop: bool = True) -> None:
        self.allow_stop = allow_stop

    def choose_action(self, env: Any) -> int:
        """
        Choose the first valid non-stop action.

        If no non-stop action is available, choose STOP if allowed.
        """

        valid_action_numbers = env.get_valid_action_numbers()

        if not valid_action_numbers:
            raise RuntimeError("No valid actions are available.")

        stop_action_number: int | None = None

        for action_number in valid_action_numbers:
            action = get_action(env.actions, action_number)

            if action.kind == ActionKind.STOP:
                stop_action_number = action_number
                continue

            return action_number

        if self.allow_stop and stop_action_number is not None:
            return stop_action_number

        raise RuntimeError("No valid non-stop actions are available.")


class GreedyCoveragePolicy:
    """
    Choose the valid action with the best immediate estimated reward.

    This policy performs a one-step lookahead:

        For each valid viewpoint:
            - estimate its viewshed
            - calculate newly visible area
            - calculate repeated visible area
            - estimate immediate reward

        Then choose the action with the highest estimated reward.

    This does not permanently update the environment. The real update still
    happens later inside env.step(action_number).
    """

    def __init__(
        self,
        *,
        stop_when_target_reached: bool = True,
        allow_stop_if_no_viewpoints: bool = True,
    ) -> None:
        self.stop_when_target_reached = stop_when_target_reached
        self.allow_stop_if_no_viewpoints = allow_stop_if_no_viewpoints

    def choose_action(self, env: Any) -> int:
        """
        Choose the valid action with the highest immediate estimated reward.
        """

        valid_action_numbers = env.get_valid_action_numbers()

        if not valid_action_numbers:
            raise RuntimeError("No valid actions are available.")

        stop_action_number = self._find_stop_action_number(env, valid_action_numbers)

        if self._should_stop_now(env, stop_action_number):
            return stop_action_number

        best_action_number: int | None = None
        best_score = float("-inf")

        for action_number in valid_action_numbers:
            action = get_action(env.actions, action_number)

            if action.kind == ActionKind.STOP:
                continue

            score = self._estimate_select_viewpoint_score(env, action_number)

            if score > best_score:
                best_score = score
                best_action_number = action_number

        if best_action_number is not None:
            return best_action_number

        if self.allow_stop_if_no_viewpoints and stop_action_number is not None:
            return stop_action_number

        raise RuntimeError("No valid viewpoint actions are available.")

    def _should_stop_now(
        self,
        env: Any,
        stop_action_number: int | None,
    ) -> bool:
        """
        Decide whether the policy should choose STOP immediately.
        """

        if stop_action_number is None:
            return False

        if not self.stop_when_target_reached:
            return False

        return env.state.coverage_percentage >= env.target_coverage_percentage

    def _find_stop_action_number(
        self,
        env: Any,
        valid_action_numbers: list[int],
    ) -> int | None:
        """
        Find the action number for STOP, if it is currently valid.
        """

        for action_number in valid_action_numbers:
            action = get_action(env.actions, action_number)

            if action.kind == ActionKind.STOP:
                return action_number

        return None

    def _estimate_select_viewpoint_score(
        self,
        env: Any,
        action_number: int,
    ) -> float:
        """
        Estimate the immediate reward for a SELECT_VIEWPOINT action.
        """

        action = get_action(env.actions, action_number)

        if action.kind != ActionKind.SELECT_VIEWPOINT:
            raise ValueError("Can only score SELECT_VIEWPOINT actions.")

        if action.viewpoint_id is None:
            raise ValueError("SELECT_VIEWPOINT action is missing viewpoint_id.")

        viewpoint = env._viewpoints_by_id[action.viewpoint_id]

        previous_visibility = env.state.visibility_mask

        if previous_visibility is None:
            raise RuntimeError("Environment visibility mask has not been initialised.")

        new_visibility = env.viewshed_function(viewpoint)
        env._validate_visibility_mask(new_visibility)

        newly_visible_mask = np.logical_and(new_visibility, ~previous_visibility)
        repeated_visible_mask = np.logical_and(new_visibility, previous_visibility)

        newly_visible_area = int(np.count_nonzero(newly_visible_mask))
        repeated_visible_area = int(np.count_nonzero(repeated_visible_mask))

        estimated_reward = env._calculate_select_viewpoint_reward(
            newly_visible_area=newly_visible_area,
            repeated_visible_area=repeated_visible_area,
            travel_cost=viewpoint.travel_cost,
            computation_cost=viewpoint.computation_cost,
        )

        updated_visibility = np.logical_or(previous_visibility, new_visibility)

        estimated_coverage = env._calculate_coverage_percentage(updated_visibility)

        if estimated_coverage >= env.target_coverage_percentage:
            estimated_reward += env.target_reached_bonus

        return estimated_reward


class EpsilonGreedyPolicy:
    """
    Mostly choose the greedy action, but sometimes choose a random action.

    This is useful for early experimentation because it gives the policy a
    chance to explore alternatives instead of always picking the apparently
    best immediate move.
    """

    def __init__(
        self,
        *,
        epsilon: float = 0.1,
        seed: int | None = None,
    ) -> None:
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("epsilon must be between 0.0 and 1.0.")

        self.epsilon = epsilon
        self.random_generator = random.Random(seed)

        self.random_policy = RandomPolicy(seed=seed)
        self.greedy_policy = GreedyCoveragePolicy()

    def choose_action(self, env: Any) -> int:
        """
        Choose a random action with probability epsilon.

        Otherwise, choose the greedy action.
        """

        should_explore = self.random_generator.random() < self.epsilon

        if should_explore:
            return self.random_policy.choose_action(env)

        return self.greedy_policy.choose_action(env)


def print_policy_choice(env: Any, policy: Policy) -> int:
    """
    Choose an action and print a readable description.

    This is useful while debugging mdp.py.
    """

    action_number = policy.choose_action(env)
    action = get_action(env.actions, action_number)

    print(f"Chosen action {action_number}: {describe_action(action)}")

    return action_number