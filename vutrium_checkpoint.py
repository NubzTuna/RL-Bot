from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


CHECKPOINT_EXTENSIONS = (".pt", ".pth")


def _expand_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else path.resolve()


def _iter_candidates(base_dir: Path) -> Iterable[Path]:
    for ext in CHECKPOINT_EXTENSIONS:
        yield from base_dir.rglob(f"*{ext}")


def resolve_checkpoint(base_file: Path, override: Optional[str] = None) -> Path:
    """Return the newest checkpoint file for Vutrium inference.

    Priority:
    1. An explicit override (CLI or env `VUTRIUM_CHECKPOINT`).
    2. A sibling `checkpoints`/`models`/`weights` directory near the script.
    3. The current working directory.
    """

    env_override = os.getenv("VUTRIUM_CHECKPOINT")
    raw_hint = override or env_override

    if raw_hint:
        hint_path = _expand_path(raw_hint)
        if hint_path.is_file():
            return hint_path
        if hint_path.is_dir():
            found = _newest_checkpoint_in(hint_path)
            if found:
                return found
            raise FileNotFoundError(f"No .pt/.pth files found under override directory: {hint_path}")
        raise FileNotFoundError(f"Checkpoint override does not exist: {hint_path}")

    # fallback: search nearby directories
    base_dir = base_file.resolve().parent
    for candidate_dir in [base_dir / "checkpoints", base_dir / "models", base_dir / "weights", Path.cwd()]:
        found = _newest_checkpoint_in(candidate_dir)
        if found:
            return found

    raise FileNotFoundError(
        "Could not find a checkpoint. Provide one via --checkpoint, the VUTRIUM_CHECKPOINT "
        "environment variable, or by placing a .pt/.pth file in a checkpoints/models/weights directory."
    )


def _newest_checkpoint_in(directory: Path) -> Optional[Path]:
    if not directory.exists() or not directory.is_dir():
        return None
    checkpoints = sorted(_iter_candidates(directory), key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0] if checkpoints else None
