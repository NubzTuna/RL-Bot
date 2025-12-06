"""Shared helpers for resolving trained policy checkpoints.

The helpers are used by both the RLBot wrapper and the Vutrium entrypoint so
the bot can find checkpoints consistently whether it is injected via DLL or run
through RLBot.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


def _iter_checkpoint_files(path: Path) -> Iterable[Path]:
    for ext in ("*.pt", "*.pth"):
        yield from path.rglob(ext)


def _newest_checkpoint(path: Path) -> Optional[Path]:
    candidates = list(_iter_checkpoint_files(path))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_checkpoint_path(anchor: Path, override: Optional[str] = None) -> str:
    """Resolve the concrete checkpoint file to load.

    Resolution order:
    1) ``override`` argument if provided.
    2) ``AERIAL_BOT_CHECKPOINT`` environment variable (file or directory).
    3) ``checkpoints`` folder next to the provided anchor file.

    The resolver prefers a file path. If given a directory, it will choose the
    newest ``.pt``/``.pth`` file recursively.
    """

    candidate = override or os.getenv("AERIAL_BOT_CHECKPOINT")

    if candidate:
        resolved = Path(candidate).expanduser()
        if resolved.is_file():
            return str(resolved.resolve())
        if resolved.is_dir():
            newest = _newest_checkpoint(resolved)
            if newest:
                return str(newest.resolve())
            raise FileNotFoundError(
                f"Configured checkpoint directory is empty: {resolved}. "
                "Place a .pt/.pth file inside."
            )
        print(
            "AERIAL_BOT_CHECKPOINT is set but does not point to a file or directory: "
            f"{resolved}. Falling back to ./checkpoints."
        )

    default_dir = (anchor.parent / "checkpoints").expanduser()
    if default_dir.exists():
        newest = _newest_checkpoint(default_dir)
        if newest:
            return str(newest.resolve())
        print(
            "Default checkpoints directory exists but is empty: "
            f"{default_dir}. Place a .pt/.pth file inside or set "
            "AERIAL_BOT_CHECKPOINT."
        )

    print(
        "No checkpoint configured. Set AERIAL_BOT_CHECKPOINT to a policy file "
        "or directory, or place .pt/.pth files in ./checkpoints next to "
        f"{anchor.name}."
    )
    # Return the canonical path users should populate; callers may still want to
    # create it or show it in logs.
    return str(default_dir.resolve())

