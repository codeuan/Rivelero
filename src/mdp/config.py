"""
Configuration values for the VISTA MDP.

This module stores experiment settings such as the maximum viewpoint budget,
target coverage threshold, reward weights, discount factor, and stopping
criteria.

Keeping these values separate makes it easier to tune the MDP without
digging through the environment, policy, or reward code.
"""


from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import rasterio
from pathlib import Path


def get_map_shape_from_dem(dem_path: Path) -> tuple[int, int]:
    """
    Read the DEM dimensions and return the shape needed for visibility masks.

    Returns:
        A tuple in the form:

            (height, width)

        where height is the number of raster rows and width is the number of
        raster columns.
    """

    if not dem_path.exists():
        raise FileNotFoundError(f"DEM file not found: {dem_path}")

    with rasterio.open(dem_path) as src:
        return src.height, src.width

@dataclass(frozen=True, slots=True)
class PathConfig:
    """
    Stores file and folder paths used by the MDP.

    Attributes:
        project_root:
            Root folder of the project.

        data_dir:
            Folder containing input data.

        results_dir:
            Folder where MDP outputs should be saved.

        dem_path:
            Path to the DEM file.

        candidate_viewpoints_path:
            Path to the candidate viewpoint CSV file.

        selected_viewpoints_output_path:
            Path where selected viewpoints may be saved.

        visibility_output_path:
            Path where the final visibility mask/map may be saved.

    This means that changes to file names can propagate across all files where the variable is mentioned.
    """

    project_root: Path = Path(".")
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")

    dem_path: Path = Path("data/dem.tif")
    candidate_viewpoints_path: Path = Path("data/candidate_viewpoints.csv")

    selected_viewpoints_output_path: Path = Path("results/selected_viewpoints.csv")
    visibility_output_path: Path = Path("results/final_visibility.npy")

    def create_output_directories(self) -> None:
        """
        Create output directories if they do not already exist.
        """

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.selected_viewpoints_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.visibility_output_path.parent.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class MDPConfig:
    """
    Stores general MDP settings.

    The map shape is not stored here because it should be read directly from
    the DEM used for the MDP run.
    """

    initial_budget: float = 100.0
    target_coverage_percentage: float = 95.0
    max_steps: int | None = 20
    include_stop_action: bool = True

    def validate(self) -> None:
        """
        Validate the MDP configuration.
        """

        if self.initial_budget < 0:
            raise ValueError("initial_budget cannot be negative.")

        if not 0 <= self.target_coverage_percentage <= 100:
            raise ValueError("target_coverage_percentage must be between 0 and 100.")

        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive or None.")


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """
    Stores reward weights used by the MDP.

    Reward idea:

        reward =
            + newly visible area
            - repeated visible area
            - travel cost
            - computation cost
            + target reached bonus
            - invalid action penalties
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

    def validate(self) -> None:
        """
        Validate reward configuration.
        """

        values = {
            "new_area_weight": self.new_area_weight,
            "repeated_area_penalty": self.repeated_area_penalty,
            "travel_cost_weight": self.travel_cost_weight,
            "computation_cost_weight": self.computation_cost_weight,
            "target_reached_bonus": self.target_reached_bonus,
            "stop_success_bonus": self.stop_success_bonus,
            "stop_early_penalty": self.stop_early_penalty,
            "invalid_action_penalty": self.invalid_action_penalty,
            "budget_failure_penalty": self.budget_failure_penalty,
        }

        for name, value in values.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative.")


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """
    Stores policy settings.

    Attributes:
        policy_name:
            Name of the policy to use.

            Suggested values:
                "random"
                "sequential"
                "greedy"
                "epsilon_greedy"

        random_seed:
            Optional seed for repeatable random behaviour.

        epsilon:
            Probability of random exploration for epsilon-greedy policy.
    """

    policy_name: str = "greedy"
    random_seed: int | None = 42
    epsilon: float = 0.1

    def validate(self) -> None:
        """
        Validate policy configuration.
        """

        valid_policy_names = {
            "random",
            "sequential",
            "greedy",
            "epsilon_greedy",
        }

        if self.policy_name not in valid_policy_names:
            raise ValueError(
                f"policy_name must be one of {sorted(valid_policy_names)}."
            )

        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be between 0.0 and 1.0.")


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """
    Stores simple logging/printing settings.
    """

    print_step_info: bool = True
    print_reward_breakdown: bool = True
    save_results: bool = True


@dataclass(frozen=True, slots=True)
class VistaMDPConfig:
    """
    Full configuration object for a VISTA MDP run.

    This groups all smaller config objects together so mdp.py can receive one
    clean config object instead of many separate values.
    """

    paths: PathConfig = PathConfig()
    mdp: MDPConfig = MDPConfig()
    rewards: RewardConfig = RewardConfig()
    policy: PolicyConfig = PolicyConfig()
    logging: LoggingConfig = LoggingConfig()

    def validate(self) -> None:
        """
        Validate all configuration sections.
        """

        self.mdp.validate()
        self.rewards.validate()
        self.policy.validate()

    def create_output_directories(self) -> None:
        """
        Create output folders used by this run.
        """

        self.paths.create_output_directories()

    def as_dict(self) -> dict[str, Any]:
        """
        Return the full configuration as a dictionary.

        Useful for debugging or saving the run settings.
        """

        return asdict(self)


def get_default_config() -> VistaMDPConfig:
    """
    Return the default VISTA MDP configuration.
    """

    config = VistaMDPConfig()
    config.validate()

    return config