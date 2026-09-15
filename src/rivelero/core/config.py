"""General configuration primitives for Rivelero.

This module contains project-wide configuration that is independent of any
particular visibility analysis.

Scientific assumptions controlling visibility -- such as observer height,
target height, field of view, maximum viewing distance, curvature correction,
or missing-metadata policies -- belong to
``rivelero.visibility.configuration.VisibilityConfiguration``.

Survey composition belongs to
``rivelero.core.configuration.ViewpointConfiguration``.

The purpose of RiveleroConfig is to centralise general processing,
reproducibility, caching, and numerical settings that may be shared by
different parts of the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os


@dataclass(slots=True)
class RiveleroConfig:
    """General runtime configuration for the Rivelero engine.

    Parameters
    ----------
    project_name
        Optional human-readable name for the current Rivelero project or
        analysis workspace.

    cache_enabled
        Whether reusable intermediate products may be cached.

    cache_directory
        Directory used for cached intermediate products. If ``None``, modules
        requiring caching may choose an appropriate project or temporary
        location.

    temporary_directory
        Optional directory for temporary processing files such as GDAL
        viewshed rasters.

    overwrite_outputs
        Whether existing output files may be overwritten by default.

    keep_temporary_files
        Whether temporary processing files should be retained after successful
        completion. Primarily useful for debugging.

    numerical_tolerance
        General floating-point comparison tolerance used where a
        module-specific tolerance has not been defined.

    random_seed
        Optional seed for stochastic or sampling procedures. Recording this
        supports reproducibility when randomized algorithms are used.

    n_workers
        Requested number of processing workers. ``None`` allows the calling
        workflow to determine an appropriate value.

    extra_settings
        Extensible dictionary for project- or application-specific runtime
        settings that are not part of the standardized Rivelero configuration.

    Notes
    -----
    This object should not be used as a container for scientific parameters.
    Parameters that alter the scientific definition of visibility or
    observability should be stored in the configuration object belonging to
    the relevant analytical module.
    """

    project_name: str | None = None

    cache_enabled: bool = True
    cache_directory: Path | str | None = None
    temporary_directory: Path | str | None = None

    overwrite_outputs: bool = False
    keep_temporary_files: bool = False

    numerical_tolerance: float = 1e-9

    random_seed: int | None = None
    n_workers: int | None = None

    extra_settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize runtime configuration."""

        self.project_name = self._optional_string(
            "project_name",
            self.project_name,
        )

        if not isinstance(self.cache_enabled, bool):
            raise TypeError("cache_enabled must be a boolean.")

        if not isinstance(self.overwrite_outputs, bool):
            raise TypeError("overwrite_outputs must be a boolean.")

        if not isinstance(self.keep_temporary_files, bool):
            raise TypeError("keep_temporary_files must be a boolean.")

        self.cache_directory = self._optional_path(
            "cache_directory",
            self.cache_directory,
        )

        self.temporary_directory = self._optional_path(
            "temporary_directory",
            self.temporary_directory,
        )

        if not isinstance(self.numerical_tolerance, (int, float)):
            raise TypeError("numerical_tolerance must be numeric.")

        self.numerical_tolerance = float(self.numerical_tolerance)

        if self.numerical_tolerance <= 0:
            raise ValueError(
                "numerical_tolerance must be greater than zero."
            )

        if self.random_seed is not None:
            if not isinstance(self.random_seed, int):
                raise TypeError("random_seed must be an integer or None.")

            if self.random_seed < 0:
                raise ValueError(
                    "random_seed must be greater than or equal to zero."
                )

        if self.n_workers is not None:
            if not isinstance(self.n_workers, int):
                raise TypeError("n_workers must be an integer or None.")

            if self.n_workers < 1:
                raise ValueError(
                    "n_workers must be greater than or equal to one."
                )

        if not isinstance(self.extra_settings, dict):
            raise TypeError("extra_settings must be a dictionary.")

    # ------------------------------------------------------------------
    # Runtime helpers
    # ------------------------------------------------------------------

    @property
    def resolved_n_workers(self) -> int:
        """Return the effective number of processing workers.

        If the user has not specified a value, the number of CPUs reported by
        the operating system is used. At least one worker is always returned.
        """

        if self.n_workers is not None:
            return self.n_workers

        return max(1, os.cpu_count() or 1)

    def ensure_directories(self) -> None:
        """Create configured cache and temporary directories when required.

        Directory creation is explicit rather than occurring automatically
        during object construction. Creating a configuration object should not
        modify the user's filesystem.
        """

        if self.cache_enabled and self.cache_directory is not None:
            self.cache_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        if self.temporary_directory is not None:
            self.temporary_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def has_cache_directory(self) -> bool:
        """Return whether an explicit cache directory has been configured."""

        return (
            self.cache_enabled
            and self.cache_directory is not None
        )

    @property
    def has_temporary_directory(self) -> bool:
        """Return whether an explicit temporary directory is configured."""

        return self.temporary_directory is not None

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _optional_string(
        name: str,
        value: str | None,
    ) -> str | None:
        """Normalize an optional non-empty string."""

        if value is None:
            return None

        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string or None.")

        result = value.strip()

        if not result:
            raise ValueError(
                f"{name} cannot be an empty string when provided."
            )

        return result

    @staticmethod
    def _optional_path(
        name: str,
        value: Path | str | None,
    ) -> Path | None:
        """Normalize an optional filesystem path."""

        if value is None:
            return None

        if not isinstance(value, (str, Path)):
            raise TypeError(
                f"{name} must be a string, pathlib.Path, or None."
            )

        if isinstance(value, str) and not value.strip():
            raise ValueError(
                f"{name} cannot be an empty string when provided."
            )

        return Path(value).expanduser()


def default_config() -> RiveleroConfig:
    """Return a default Rivelero runtime configuration.

    This helper provides an explicit way for applications and the GUI to
    request the standard runtime configuration without maintaining their own
    copies of default values.
    """

    return RiveleroConfig()