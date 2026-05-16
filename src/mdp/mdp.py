"""
Run the VISTA Markov Decision Process.

This module coordinates a complete MDP episode by creating the environment,
resetting its initial state, selecting actions through a policy, stepping the
environment forward, and recording the resulting rewards and selected
viewpoints.

It should act as the entry point for experiments, not as the place where
viewshed calculations, reward logic, state updates, or action definitions are
implemented.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from actions import describe_action, get_action
from config import VistaMDPConfig, get_default_config
from environment import CandidateViewpoint, VistaEnvironment
from policy import (
    EpsilonGreedyPolicy,
    GreedyCoveragePolicy,
    RandomPolicy,
    SequentialPolicy,
)


def load_candidate_viewpoints(csv_path: Path) -> list[CandidateViewpoint]:
    """
    Load candidate viewpoints from a CSV file.

    Expected columns:
        viewpoint_id
        x
        y

    Optional columns:
        travel_cost
        computation_cost

    Example CSV:

        viewpoint_id,x,y,travel_cost,computation_cost
        0,120.5,340.2,1.0,1.0
        1,125.0,345.7,2.0,1.0
        2,130.2,348.1,1.5,1.0

    Args:
        csv_path:
            Path to the candidate viewpoint CSV file.

    Returns:
        A list of CandidateViewpoint objects.
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"Candidate viewpoint file not found: {csv_path}")

    candidate_viewpoints: list[CandidateViewpoint] = []

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        required_columns = {"viewpoint_id", "x", "y"}

        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header row: {csv_path}")

        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                f"Candidate viewpoint CSV is missing columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            viewpoint = CandidateViewpoint(
                viewpoint_id=int(row["viewpoint_id"]),
                x=float(row["x"]),
                y=float(row["y"]),
                travel_cost=float(row.get("travel_cost", 1.0) or 1.0),
                computation_cost=float(row.get("computation_cost", 1.0) or 1.0),
            )

            candidate_viewpoints.append(viewpoint)

    if not candidate_viewpoints:
        raise ValueError(f"No candidate viewpoints were loaded from: {csv_path}")

    return candidate_viewpoints


def fake_viewshed_function_factory(
    map_shape: tuple[int, int],
):
    """
    Create a temporary fake viewshed function for testing.

    This exists so the MDP loop can be tested before the real GDAL/rasterio
    viewshed code is connected.

    The fake viewshed creates a simple horizontal and vertical band based on
    the viewpoint ID.

    Replace this later with your real visibility function.
    """

    height, width = map_shape

    def fake_viewshed_function(viewpoint: CandidateViewpoint) -> np.ndarray:
        mask = np.zeros(map_shape, dtype=bool)

        row = viewpoint.viewpoint_id % height
        col = viewpoint.viewpoint_id % width

        mask[row, :] = True
        mask[:, col] = True

        return mask

    return fake_viewshed_function


def create_policy(config: VistaMDPConfig):
    """
    Create the policy selected in config.py.

    Returns:
        A policy object with a choose_action(env) method.
    """

    policy_name = config.policy.policy_name

    if policy_name == "random":
        return RandomPolicy(seed=config.policy.random_seed)

    if policy_name == "sequential":
        return SequentialPolicy()

    if policy_name == "greedy":
        return GreedyCoveragePolicy()

    if policy_name == "epsilon_greedy":
        return EpsilonGreedyPolicy(
            epsilon=config.policy.epsilon,
            seed=config.policy.random_seed,
        )

    raise ValueError(f"Unknown policy name: {policy_name}")


def create_environment(
    *,
    config: VistaMDPConfig,
    candidate_viewpoints: list[CandidateViewpoint],
) -> VistaEnvironment:
    """
    Create the VISTA MDP environment.

    For now, this uses a fake viewshed function so that the MDP pipeline can
    be tested before the real visibility function is connected.
    """

    viewshed_function = fake_viewshed_function_factory(
        map_shape=config.mdp.map_shape,
    )

    env = VistaEnvironment(
        candidate_viewpoints=candidate_viewpoints,
        viewshed_function=viewshed_function,
        map_shape=config.mdp.map_shape,
        initial_budget=config.mdp.initial_budget,
        target_coverage_percentage=config.mdp.target_coverage_percentage,
        max_steps=config.mdp.max_steps,
        new_area_weight=config.rewards.new_area_weight,
        repeated_area_penalty=config.rewards.repeated_area_penalty,
        travel_cost_weight=config.rewards.travel_cost_weight,
        computation_cost_weight=config.rewards.computation_cost_weight,
        target_reached_bonus=config.rewards.target_reached_bonus,
        stop_early_penalty=config.rewards.stop_early_penalty,
    )

    return env


def run_mdp(
    *,
    config: VistaMDPConfig,
    candidate_viewpoints: list[CandidateViewpoint],
) -> dict[str, Any]:
    """
    Run one complete MDP episode.

    Args:
        config:
            Full MDP configuration object.

        candidate_viewpoints:
            Candidate viewpoints available to the MDP.

    Returns:
        A dictionary containing summary results.
    """

    env = create_environment(
        config=config,
        candidate_viewpoints=candidate_viewpoints,
    )

    policy = create_policy(config)

    state = env.reset()

    done = False
    total_reward = 0.0
    step_results: list[dict[str, Any]] = []

    if config.logging.print_step_info:
        print("Starting VISTA MDP run")
        print(f"Policy: {config.policy.policy_name}")
        print(f"Initial budget: {config.mdp.initial_budget}")
        print(f"Target coverage: {config.mdp.target_coverage_percentage}%")
        print()

    while not done:
        action_number = policy.choose_action(env)
        action = get_action(env.actions, action_number)

        state, reward, done, info = env.step(action_number)

        total_reward += reward

        step_record = {
            "step": state.steps_taken,
            "action_number": action_number,
            "action_description": describe_action(action),
            "reward": reward,
            "total_reward": total_reward,
            "coverage_percentage": state.coverage_percentage,
            "redundancy_score": state.redundancy_score,
            "remaining_budget": state.remaining_budget,
            "done": done,
            "info": make_json_safe(info),
        }

        step_results.append(step_record)

        if config.logging.print_step_info:
            print(f"Step {state.steps_taken}")
            print(f"Action: {action_number} - {describe_action(action)}")
            print(f"Reward: {reward}")
            print(f"Total reward: {total_reward}")
            print(f"Coverage: {state.coverage_percentage:.2f}%")
            print(f"Redundancy: {state.redundancy_score}")
            print(f"Remaining budget: {state.remaining_budget}")
            print(f"Reason: {info.get('reason', 'no reason given')}")
            print()

    summary = {
        "total_reward": total_reward,
        "steps_taken": state.steps_taken,
        "selected_viewpoints": sorted(state.selected_viewpoints),
        "coverage_percentage": state.coverage_percentage,
        "redundancy_score": state.redundancy_score,
        "remaining_budget": state.remaining_budget,
        "step_results": step_results,
    }

    if config.logging.print_step_info:
        print("Finished VISTA MDP run")
        print(f"Selected viewpoints: {summary['selected_viewpoints']}")
        print(f"Final coverage: {summary['coverage_percentage']:.2f}%")
        print(f"Total reward: {summary['total_reward']}")

    if config.logging.save_results:
        save_results(
            config=config,
            summary=summary,
            final_visibility_mask=state.visibility_mask,
        )

    return summary


def save_results(
    *,
    config: VistaMDPConfig,
    summary: dict[str, Any],
    final_visibility_mask: np.ndarray | None,
) -> None:
    """
    Save MDP results to the results folder.
    """

    config.create_output_directories()

    save_selected_viewpoints(
        output_path=config.paths.selected_viewpoints_output_path,
        selected_viewpoints=summary["selected_viewpoints"],
    )

    save_summary_json(
        output_path=config.paths.results_dir / "mdp_summary.json",
        summary=summary,
    )

    if final_visibility_mask is not None:
        np.save(config.paths.visibility_output_path, final_visibility_mask)


def save_selected_viewpoints(
    *,
    output_path: Path,
    selected_viewpoints: list[int],
) -> None:
    """
    Save selected viewpoint IDs to a CSV file.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["selection_order", "viewpoint_id"])
        writer.writeheader()

        for selection_order, viewpoint_id in enumerate(selected_viewpoints, start=1):
            writer.writerow(
                {
                    "selection_order": selection_order,
                    "viewpoint_id": viewpoint_id,
                }
            )


def save_summary_json(
    *,
    output_path: Path,
    summary: dict[str, Any],
) -> None:
    """
    Save the MDP summary to a JSON file.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(
            make_json_safe(summary),
            json_file,
            indent=4,
        )


def make_json_safe(value: Any) -> Any:
    """
    Convert common non-JSON-safe values into JSON-safe values.

    This helps when saving dictionaries that contain sets, NumPy numbers,
    NumPy arrays, or Path objects.
    """

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [make_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]

    if isinstance(value, set):
        return sorted(make_json_safe(item) for item in value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    return value


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.
    """

    parser = argparse.ArgumentParser(
        description="Run the VISTA viewpoint-selection MDP.",
    )

    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="Optional path to candidate_viewpoints.csv.",
    )

    parser.add_argument(
        "--policy",
        type=str,
        choices=["random", "sequential", "greedy", "epsilon_greedy"],
        default=None,
        help="Optional policy override.",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the MDP without saving output files.",
    )

    return parser


def apply_cli_overrides(
    *,
    config: VistaMDPConfig,
    args: argparse.Namespace,
) -> VistaMDPConfig:
    """
    Apply simple command-line overrides to the config.

    Since the config dataclasses are frozen, this creates a new config object
    when overrides are needed.
    """

    from dataclasses import replace

    paths = config.paths
    policy = config.policy
    logging_config = config.logging

    if args.candidates is not None:
        paths = replace(
            paths,
            candidate_viewpoints_path=args.candidates,
        )

    if args.policy is not None:
        policy = replace(
            policy,
            policy_name=args.policy,
        )

    if args.no_save:
        logging_config = replace(
            logging_config,
            save_results=False,
        )

    updated_config = replace(
        config,
        paths=paths,
        policy=policy,
        logging=logging_config,
    )

    updated_config.validate()

    return updated_config


def main() -> None:
    """
    Main entry point for command-line execution.
    """

    parser = build_argument_parser()
    args = parser.parse_args()

    config = get_default_config()
    config = apply_cli_overrides(config=config, args=args)

    candidate_viewpoints = load_candidate_viewpoints(
        config.paths.candidate_viewpoints_path,
    )

    run_mdp(
        config=config,
        candidate_viewpoints=candidate_viewpoints,
    )


if __name__ == "__main__":
    main()