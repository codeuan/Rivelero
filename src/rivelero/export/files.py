"""Atomic single-file writing shared by all exporters."""

from __future__ import annotations

import os
import re
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4


@contextmanager
def atomic_output(path: Path | str, *, overwrite: bool = False) -> Iterator[Path]:
    """Yield a temporary path that replaces ``path`` only on success.

    The temporary file lives next to the destination so the final
    ``os.replace`` is atomic. If writing raises, the temporary file is
    removed and any existing destination is left untouched. An existing
    destination is refused unless ``overwrite`` is True.
    """

    target = Path(path).expanduser().resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {target}")
    check_path_length(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Short name: the temporary must never be the path that exceeds limits.
    temporary = target.with_name(f".{uuid4().hex[:12]}.tmp{target.suffix}")
    try:
        yield temporary
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        # GDAL may write auxiliary files next to the temporary raster.
        for auxiliary in target.parent.glob(f"{temporary.name}.*"):
            auxiliary.unlink(missing_ok=True)
        raise


# Classic Windows MAX_PATH (260 including the terminator). Paths beyond it
# fail inside GDAL and os.replace unless long-path support is enabled, so an
# export refuses them up front instead of failing half-way.
MAX_PATH_LENGTH = 259 if os.name == "nt" else None


class PathTooLongError(OSError):
    """Raised before writing when a destination path is too long."""


def check_path_length(path: Path | str) -> None:
    if MAX_PATH_LENGTH is None:
        return
    text = str(Path(path).expanduser().resolve())
    if len(text) > MAX_PATH_LENGTH:
        raise PathTooLongError(
            f"The path is {len(text)} characters long; Windows allows "
            f"{MAX_PATH_LENGTH}. Choose a shorter folder or project name: {text}"
        )


def safe_filename_part(value: str, *, fallback: str = "untitled") -> str:
    """Turn a user/project/scenario name into a portable filename part."""

    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return (text or fallback)[:80]
