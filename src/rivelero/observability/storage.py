"""Visibility storage and caching for Rivelero.

This module manages the relationship between sampling units (Viewpoints or
ObservationEvents) and their computed visibility masks.

The storage architecture follows four principles:

1. Viewpoints remain lightweight scientific metadata objects.
2. Visibility is computed lazily, only when requested.
3. Computed visibility masks are persisted on disk for reuse.
4. Recently accessed masks are retained in an in-memory LRU cache.

The SurveyObservabilityField and downstream metrics should interact with a
VisibilityStore rather than depending directly on the physical storage format.

The initial backend uses compressed NumPy NPZ files. This is intentionally an
implementation detail: future backends may use sparse arrays, bit-packed
masks, Zarr, databases, or other storage systems without changing the
higher-level observability architecture.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterator
import json
import os
import tempfile

import numpy as np
from affine import Affine
from rasterio.crs import CRS

from rivelero.visibility.engine import SingleViewpointVisibility


# ---------------------------------------------------------------------------
# Cache identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VisibilityKey:
    """Unique identity of one cached visibility calculation.

    A visibility mask is not determined by a Viewpoint alone. It also depends
    on the Environment, AnalysisDomain, VisibilityConfiguration, and whether
    the sampling unit is a Viewpoint or an ObservationEvent.

    Parameters
    ----------
    sampling_unit_id
        Identifier of the Viewpoint or ObservationEvent being evaluated.

    sampling_unit_type
        Normally ``"viewpoint"`` or ``"observation_event"``.

    viewpoint_id
        Identifier of the underlying Viewpoint.

    environment_id
        Environment used for the calculation.

    analysis_domain_id
        AnalysisDomain used for the calculation.

    visibility_configuration_id
        VisibilityConfiguration used for the calculation.
    """

    sampling_unit_id: str
    sampling_unit_type: str

    viewpoint_id: str

    environment_id: str
    analysis_domain_id: str
    visibility_configuration_id: str

    def __post_init__(self) -> None:
        for name in (
            "sampling_unit_id",
            "sampling_unit_type",
            "viewpoint_id",
            "environment_id",
            "analysis_domain_id",
            "visibility_configuration_id",
        ):
            value = getattr(self, name)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")

    @property
    def canonical_string(self) -> str:
        """Return a stable textual representation of the cache identity."""

        return "|".join(
            (
                self.sampling_unit_type,
                self.sampling_unit_id,
                self.viewpoint_id,
                self.environment_id,
                self.analysis_domain_id,
                self.visibility_configuration_id,
            )
        )

    @property
    def digest(self) -> str:
        """Return a stable SHA-256 digest suitable for filenames."""

        return sha256(
            self.canonical_string.encode("utf-8")
        ).hexdigest()


# ---------------------------------------------------------------------------
# Stored visibility representation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StoredVisibility:
    """Lightweight visibility object used by the storage layer.

    This deliberately stores the final effective visibility mask rather than
    the complete SingleViewpointVisibility result.

    The geometric pre-directional mask may be useful for diagnostics but is
    not required for the fundamental SOF viewpoint-cell relationship.
    """

    key: VisibilityKey

    visibility_mask: np.ndarray

    transform: Affine
    crs: CRS

    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.visibility_mask, np.ndarray):
            raise TypeError(
                "visibility_mask must be a numpy.ndarray."
            )

        if self.visibility_mask.ndim != 2:
            raise ValueError(
                "visibility_mask must be two-dimensional."
            )

        self.visibility_mask = self.visibility_mask.astype(
            bool,
            copy=False,
        )

        if not isinstance(self.transform, Affine):
            raise TypeError("transform must be an Affine.")

        if not isinstance(self.crs, CRS):
            self.crs = CRS.from_user_input(self.crs)

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")

    @property
    def shape(self) -> tuple[int, int]:
        """Return visibility raster shape."""

        return self.visibility_mask.shape

    @property
    def visible_cell_count(self) -> int:
        """Return number of visible cells."""

        return int(np.count_nonzero(self.visibility_mask))


# ---------------------------------------------------------------------------
# Visibility store
# ---------------------------------------------------------------------------


class VisibilityStore:
    """Lazy disk-backed visibility store with an in-memory LRU cache.

    Parameters
    ----------
    cache_directory
        Directory used to persist calculated visibility masks.

    max_memory_items
        Maximum number of visibility masks retained simultaneously in the
        in-memory LRU cache.

        This limit is item-based rather than byte-based in the initial
        implementation. A byte-aware cache can be introduced later without
        changing the public storage interface.

    compressed
        Whether NPZ files should use NumPy compression.

    Notes
    -----
    The store does not itself know how to calculate visibility. A computation
    callback is supplied to ``get_or_compute`` when a cache miss occurs.

    This keeps the storage layer independent of Viewpoint, Environment, and
    VisibilityConfiguration construction logic.
    """

    STORAGE_VERSION = 1

    def __init__(
        self,
        cache_directory: Path | str,
        *,
        max_memory_items: int = 32,   # Limit of 32 is arbitrary; can be tuned for performance. Eventually this should probably become memory-size based, for example:maximum RAM cache = 1 GB
        compressed: bool = True,
    ) -> None:
        self.cache_directory = Path(
            cache_directory
        ).expanduser().resolve()

        if not isinstance(max_memory_items, int):
            raise TypeError(
                "max_memory_items must be an integer."
            )

        if max_memory_items < 0:
            raise ValueError(
                "max_memory_items must be greater than or equal to zero."
            )

        if not isinstance(compressed, bool):
            raise TypeError("compressed must be a boolean.")

        self.max_memory_items = max_memory_items
        self.compressed = compressed

        self.cache_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # digest -> StoredVisibility
        self._memory_cache: OrderedDict[
            str,
            StoredVisibility,
        ] = OrderedDict()

    # ------------------------------------------------------------------
    # Public retrieval API
    # ------------------------------------------------------------------

    def get(
        self,
        key: VisibilityKey,
    ) -> StoredVisibility | None:
        """Retrieve visibility from RAM or disk.

        Returns None when the requested visibility has never been calculated
        or is no longer present on disk.
        """

        self._validate_key(key)

        digest = key.digest

        # Fastest path: in-memory LRU cache.
        if digest in self._memory_cache:
            result = self._memory_cache.pop(digest)

            # Reinsert as most recently used.
            self._memory_cache[digest] = result

            return result

        # Second level: persistent disk cache.
        path = self.path_for(key)

        if not path.is_file():
            return None

        result = self._load_from_disk(
            key=key,
            path=path,
        )

        self._put_memory(result)

        return result

    def get_or_compute(
        self,
        key: VisibilityKey,
        compute: Callable[[], SingleViewpointVisibility],
    ) -> StoredVisibility:
        """Retrieve visibility or calculate it lazily on a cache miss.

        Parameters
        ----------
        key
            Identity of the requested visibility calculation.

        compute
            Zero-argument callback that calculates and returns a
            SingleViewpointVisibility if the result is not already cached.

        Returns
        -------
        StoredVisibility
            Cached or newly calculated effective visibility.

        Notes
        -----
        ``compute`` is called only when neither RAM nor disk contains the
        requested result.
        """

        existing = self.get(key)

        if existing is not None:
            return existing

        if not callable(compute):
            raise TypeError("compute must be callable.")

        result = compute()

        if not isinstance(result, SingleViewpointVisibility):
            raise TypeError(
                "compute must return SingleViewpointVisibility."
            )

        self._validate_computed_result(
            key=key,
            result=result,
        )

        stored = StoredVisibility(
            key=key,
            visibility_mask=result.visibility_mask.copy(),
            transform=result.transform,
            crs=result.crs,
            metadata={
                **result.metadata,
                "event_id": result.event_id,
                "visible_cell_count": result.visible_cell_count,
            },
        )

        self.put(stored)

        return stored

    def put(
        self,
        visibility: StoredVisibility,
    ) -> Path:
        """Persist a visibility result and place it in the LRU cache."""

        if not isinstance(visibility, StoredVisibility):
            raise TypeError(
                "visibility must be a StoredVisibility."
            )

        path = self.path_for(visibility.key)

        self._write_to_disk_atomic(
            visibility=visibility,
            path=path,
        )

        self._put_memory(visibility)

        return path

    # ------------------------------------------------------------------
    # Existence and path queries
    # ------------------------------------------------------------------

    def contains(
        self,
        key: VisibilityKey,
    ) -> bool:
        """Return whether a result exists in RAM or on disk."""

        self._validate_key(key)

        return (
            key.digest in self._memory_cache
            or self.path_for(key).is_file()
        )

    def path_for(
        self,
        key: VisibilityKey,
    ) -> Path:
        """Return the disk-cache path associated with a key."""

        self._validate_key(key)

        # Two-character prefix prevents very large caches from placing every
        # file in one filesystem directory.
        prefix = key.digest[:2]

        directory = self.cache_directory / prefix

        return directory / f"{key.digest}.npz"

    # ------------------------------------------------------------------
    # Removal / invalidation
    # ------------------------------------------------------------------

    def remove(
        self,
        key: VisibilityKey,
        *,
        remove_disk: bool = True,
    ) -> bool:
        """Remove a cached visibility result.

        Parameters
        ----------
        key
            Visibility result to remove.

        remove_disk
            If True, delete the persistent disk copy as well as the RAM copy.

        Returns
        -------
        bool
            True if at least one cached representation was removed.
        """

        self._validate_key(key)

        removed = False

        if self._memory_cache.pop(key.digest, None) is not None:
            removed = True

        if remove_disk:
            path = self.path_for(key)

            if path.is_file():
                path.unlink()
                removed = True

        return removed

    def clear_memory(self) -> None:
        """Empty the in-memory LRU cache without deleting disk data."""

        self._memory_cache.clear()

    def clear_disk(self) -> int:
        """Delete all visibility files managed by this store.

        Returns
        -------
        int
            Number of NPZ files deleted.

        Notes
        -----
        The cache root directory itself is retained.
        """

        count = 0

        for path in self.cache_directory.rglob("*.npz"):
            path.unlink()
            count += 1

        self.clear_memory()

        return count

    # ------------------------------------------------------------------
    # Cache inspection
    # ------------------------------------------------------------------

    @property
    def memory_item_count(self) -> int:
        """Number of visibility masks currently resident in RAM."""

        return len(self._memory_cache)

    @property
    def memory_keys(self) -> tuple[str, ...]:
        """Return cache digests from least to most recently used."""

        return tuple(self._memory_cache.keys())

    def iter_disk_paths(self) -> Iterator[Path]:
        """Iterate over persistent visibility files."""

        yield from self.cache_directory.rglob("*.npz")

    @property
    def disk_item_count(self) -> int:
        """Return number of persisted visibility results."""

        return sum(1 for _ in self.iter_disk_paths())

    # ------------------------------------------------------------------
    # LRU implementation
    # ------------------------------------------------------------------

    def _put_memory(
        self,
        visibility: StoredVisibility,
    ) -> None:
        """Insert visibility into RAM and enforce LRU capacity."""

        if self.max_memory_items == 0:
            return

        digest = visibility.key.digest

        # Remove old instance so reinsertion marks it as most recently used.
        self._memory_cache.pop(digest, None)

        self._memory_cache[digest] = visibility

        while len(self._memory_cache) > self.max_memory_items:
            # OrderedDict stores least recently used at the beginning.
            self._memory_cache.popitem(last=False)

    # ------------------------------------------------------------------
    # Disk serialization
    # ------------------------------------------------------------------

    def _write_to_disk_atomic(
        self,
        *,
        visibility: StoredVisibility,
        path: Path,
    ) -> None:
        """Write one visibility result atomically."""

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata = {
            "storage_version": self.STORAGE_VERSION,
            "key": {
                "sampling_unit_id": visibility.key.sampling_unit_id,
                "sampling_unit_type": visibility.key.sampling_unit_type,
                "viewpoint_id": visibility.key.viewpoint_id,
                "environment_id": visibility.key.environment_id,
                "analysis_domain_id": visibility.key.analysis_domain_id,
                "visibility_configuration_id": (
                    visibility.key.visibility_configuration_id
                ),
            },
            "transform": tuple(visibility.transform)[:6],
            "crs": visibility.crs.to_string(),
            "shape": list(visibility.shape),
            "metadata": self._json_safe_mapping(
                visibility.metadata
            ),
        }

        # A temporary file in the destination directory permits atomic
        # replacement on the same filesystem.
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{visibility.key.digest}.",
            suffix=".npz",
            dir=path.parent,
        )

        os.close(fd)

        temporary_path = Path(temporary_name)

        try:
            if self.compressed:
                np.savez_compressed(
                    temporary_path,
                    visibility_mask=visibility.visibility_mask,
                    metadata_json=np.array(
                        json.dumps(metadata),
                        dtype=np.str_,
                    ),
                )
            else:
                np.savez(
                    temporary_path,
                    visibility_mask=visibility.visibility_mask,
                    metadata_json=np.array(
                        json.dumps(metadata),
                        dtype=np.str_,
                    ),
                )

            os.replace(
                temporary_path,
                path,
            )

        except Exception:
            temporary_path.unlink(
                missing_ok=True,
            )
            raise

    def _load_from_disk(
        self,
        *,
        key: VisibilityKey,
        path: Path,
    ) -> StoredVisibility:
        """Load and validate a persistent visibility result."""

        try:
            with np.load(
                path,
                allow_pickle=False,
            ) as archive:
                visibility_mask = archive[
                    "visibility_mask"
                ].astype(bool, copy=False)

                metadata_json = str(
                    archive["metadata_json"].item()
                )

        except Exception as exc:
            raise RuntimeError(
                f"Could not load visibility cache file: {path}"
            ) from exc

        metadata = json.loads(metadata_json)

        if metadata.get("storage_version") != self.STORAGE_VERSION:
            raise RuntimeError(
                f"Unsupported visibility-cache version in {path}."
            )

        stored_key = metadata.get("key", {})

        expected_key = {
            "sampling_unit_id": key.sampling_unit_id,
            "sampling_unit_type": key.sampling_unit_type,
            "viewpoint_id": key.viewpoint_id,
            "environment_id": key.environment_id,
            "analysis_domain_id": key.analysis_domain_id,
            "visibility_configuration_id": (
                key.visibility_configuration_id
            ),
        }

        if stored_key != expected_key:
            raise RuntimeError(
                f"Visibility cache key mismatch in {path}."
            )

        expected_shape = tuple(
            metadata.get("shape", ())
        )

        if visibility_mask.shape != expected_shape:
            raise RuntimeError(
                f"Visibility cache shape mismatch in {path}."
            )

        transform_values = metadata.get("transform")

        if (
            not isinstance(transform_values, list)
            or len(transform_values) != 6
        ):
            raise RuntimeError(
                f"Invalid affine transform metadata in {path}."
            )

        transform = Affine(*transform_values)

        crs_value = metadata.get("crs")

        if not crs_value:
            raise RuntimeError(
                f"Missing CRS metadata in {path}."
            )

        return StoredVisibility(
            key=key,
            visibility_mask=visibility_mask,
            transform=transform,
            crs=CRS.from_user_input(crs_value),
            metadata=metadata.get("metadata", {}),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_key(
        key: VisibilityKey,
    ) -> None:
        if not isinstance(key, VisibilityKey):
            raise TypeError("key must be a VisibilityKey.")

    @staticmethod
    def _validate_computed_result(
        *,
        key: VisibilityKey,
        result: SingleViewpointVisibility,
    ) -> None:
        """Ensure a computed result corresponds to the requested key."""

        if result.viewpoint_id != key.viewpoint_id:
            raise ValueError(
                "Computed visibility viewpoint_id does not match "
                "VisibilityKey."
            )

        if key.sampling_unit_type == "observation_event":
            if result.event_id != key.sampling_unit_id:
                raise ValueError(
                    "Computed visibility event_id does not match "
                    "VisibilityKey."
                )

        elif key.sampling_unit_type == "viewpoint":
            if key.sampling_unit_id != key.viewpoint_id:
                raise ValueError(
                    "For viewpoint sampling, sampling_unit_id must equal "
                    "viewpoint_id."
                )

        else:
            raise ValueError(
                "sampling_unit_type must currently be 'viewpoint' or "
                "'observation_event'."
            )

    @classmethod
    def _json_safe_mapping(
        cls,
        mapping: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert common metadata values to JSON-safe representations."""

        result: dict[str, Any] = {}

        for key, value in mapping.items():
            if value is None or isinstance(
                value,
                (str, int, float, bool),
            ):
                result[str(key)] = value

            elif isinstance(value, Path):
                result[str(key)] = str(value)

            elif isinstance(value, CRS):
                result[str(key)] = value.to_string()

            elif isinstance(value, (list, tuple)):
                result[str(key)] = [
                    cls._json_safe_value(item)
                    for item in value
                ]

            elif isinstance(value, dict):
                result[str(key)] = cls._json_safe_mapping(
                    value
                )

            else:
                # Preserve provenance without permitting arbitrary pickle data.
                result[str(key)] = str(value)

        return result

    @classmethod
    def _json_safe_value(
        cls,
        value: Any,
    ) -> Any:
        if value is None or isinstance(
            value,
            (str, int, float, bool),
        ):
            return value

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, CRS):
            return value.to_string()

        if isinstance(value, dict):
            return cls._json_safe_mapping(value)

        if isinstance(value, (list, tuple)):
            return [
                cls._json_safe_value(item)
                for item in value
            ]

        return str(value)