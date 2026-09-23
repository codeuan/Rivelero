"""Project format identity, schema versioning and migrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


PROJECT_FORMAT = "rivelero-project"

# Version of the *project file layout*, independent of the software version.
# Increment when a saved document changes incompatibly and add a migration.
SCHEMA_VERSION = 1

PROJECT_SUFFIX = ".rivelero"


class ProjectFormatError(ValueError):
    """The file is not a valid Rivelero project."""


class UnsupportedSchemaError(ProjectFormatError):
    """The project uses a schema version this software cannot read."""


class ProjectSerializationError(ValueError):
    """A value cannot be represented safely in a project."""


# Migrations upgrade the documents of schema version N to N + 1:
#   MIGRATIONS[N](documents) -> documents
# None exist yet; the loader applies them in sequence before decoding.
MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def check_manifest(manifest: Any) -> int:
    """Validate format identity and return the project's schema version."""

    if not isinstance(manifest, dict) or manifest.get("format") != PROJECT_FORMAT:
        raise ProjectFormatError("This file is not a Rivelero project.")

    version = manifest.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ProjectFormatError("The project has no valid schema_version.")

    if version > SCHEMA_VERSION:
        raise UnsupportedSchemaError(
            f"The project uses schema version {version}, but this version of "
            f"Rivelero reads up to version {SCHEMA_VERSION}. Update Rivelero "
            "to open it."
        )
    return version


def migrate(documents: dict[str, Any], version: int) -> dict[str, Any]:
    """Upgrade documents from ``version`` to the current schema."""

    while version < SCHEMA_VERSION:
        step = MIGRATIONS.get(version)
        if step is None:
            raise UnsupportedSchemaError(
                f"No migration from schema version {version} is available."
            )
        documents = step(documents)
        version += 1
    return documents


def software_metadata() -> dict[str, Any]:
    """Best-effort software identity recorded in each saved project."""

    from rivelero import __version__

    return {
        "name": "rivelero",
        "version": __version__,
        "git_commit": _git_commit(),
    }


def _git_commit() -> str | None:
    """Read the current commit from a source checkout without running git."""

    for parent in Path(__file__).resolve().parents:
        git = parent / ".git"
        if not git.is_dir():
            continue
        try:
            head = (git / "HEAD").read_text(encoding="utf-8").strip()
            if head.startswith("ref: "):
                ref = git / head[5:]
                if ref.is_file():
                    return ref.read_text(encoding="utf-8").strip()
                packed = git / "packed-refs"
                if packed.is_file():
                    for line in packed.read_text(encoding="utf-8").splitlines():
                        if line.endswith(head[5:]):
                            return line.split()[0]
                return None
            return head or None
        except OSError:
            return None
    return None
