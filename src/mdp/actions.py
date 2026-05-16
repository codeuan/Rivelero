"""
Action definitions for the VISTA MDP.

This module defines the choices available to the agent. In the simplest
version, the agent can either select a candidate viewpoint or stop the
search.

Keeping actions separate makes it easier to add new decision types later,
such as moving to a nearby location, changing viewing direction, scanning
from the current point, or increasing the search radius.
"""


from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class ActionKind(str, Enum):
    """
    Actions available in a given state are classified under these.
    """

    SELECT_VIEWPOINT = "select_viewpoint"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class Action:
    """
    Represents one action available to the MDP agent.

    Attributes:
        kind:
            The type of action being taken.

        viewpoint_id:
            The candidate viewpoint selected by the action.

            This is required for SELECT_VIEWPOINT actions and should be None
            for STOP actions.
            
    frozen is True to stop actions from being changed later.

    slots enforces that only the fields declared are allowed to be used.
    """

    kind: ActionKind
    viewpoint_id: int | None = None

    def __post_init__(self) -> None:
        """
        Validate that the action contains sensible data.
        """

        if self.kind == ActionKind.SELECT_VIEWPOINT and self.viewpoint_id is None:
            raise ValueError("SELECT_VIEWPOINT actions require a viewpoint_id.")

        if self.kind == ActionKind.STOP and self.viewpoint_id is not None:
            raise ValueError("STOP actions should not have a viewpoint_id.")


def make_select_viewpoint_action(viewpoint_id: int) -> Action:
    """
    Create an action that selects a candidate viewpoint.

    Args:
        viewpoint_id:
            The ID/index of the candidate viewpoint.

    Returns:
        An Action representing the selection of that viewpoint.
    """

    return Action(
        kind=ActionKind.SELECT_VIEWPOINT,
        viewpoint_id=viewpoint_id,
    )


def make_stop_action() -> Action:
    """
    Create the action used to stop the MDP episode.

    Returns:
        An Action representing the decision to stop.
    """

    return Action(kind=ActionKind.STOP)


def build_actions(
    candidate_viewpoint_ids: Iterable[int],
) -> list[Action]:
    """
    Build the full list of actions available to the MDP agent.

    Each candidate viewpoint becomes one SELECT_VIEWPOINT action.
    Optionally, a STOP action is added at the end.

    Args:
        candidate_viewpoint_ids:
            The IDs/indices of the candidate viewpoints.

        include_stop:
            Whether to include a STOP action.

    Returns:
        A list of Action objects.

    Example:
        If candidate_viewpoint_ids = [0, 1, 2], this returns:

            [
                Action(kind=SELECT_VIEWPOINT, viewpoint_id=0),
                Action(kind=SELECT_VIEWPOINT, viewpoint_id=1),
                Action(kind=SELECT_VIEWPOINT, viewpoint_id=2),
                Action(kind=STOP),
            ]
    """

    actions: list[Action] = []

    for viewpoint_id in candidate_viewpoint_ids:
        actions.append(make_select_viewpoint_action(viewpoint_id))


    actions.append(make_stop_action())

    return actions


def get_action(actions: list[Action], action_number: int) -> Action:
    """
    Convert an integer action number into an Action object.

    The policy will usually choose an integer. This function translates that
    integer into the actual action description.

    Args:
        actions:
            The full list of available actions.

        action_number:
            The integer chosen by the policy.

    Returns:
        The Action at that position in the action list.

    Raises:
        IndexError:
            If the action number is outside the valid range.
    """

    if action_number < 0 or action_number >= len(actions):
        raise IndexError(
            f"Action number {action_number} is outside the valid range "
            f"0 to {len(actions) - 1}."
        )

    return actions[action_number]


def get_action_number(actions: list[Action], action: Action) -> int:
    """
    Convert an Action object back into its integer action number.

    This is mainly useful for testing, logging, and debugging.

    Args:
        actions:
            The full list of available actions.

        action:
            The Action object to look up.

    Returns:
        The integer position of the action.

    Raises:
        ValueError:
            If the action is not in the action list.
    """

    try:
        return actions.index(action)
    except ValueError as exc:
        raise ValueError("The given action is not in the action list.") from exc


def is_select_viewpoint_action(action: Action) -> bool:
    """
    Return True if the action selects a candidate viewpoint.
    """

    return action.kind == ActionKind.SELECT_VIEWPOINT


def is_stop_action(action: Action) -> bool:
    """
    Return True if the action stops the episode.
    """

    return action.kind == ActionKind.STOP


def describe_action(action: Action) -> str:
    """
    Return a human-readable description of an action.

    This is useful for printing progress inside mdp.py.
    """

    if action.kind == ActionKind.SELECT_VIEWPOINT:
        return f"Select viewpoint {action.viewpoint_id}"

    if action.kind == ActionKind.STOP:
        return "Stop episode"

    return "Unknown action"