"""Shared helpers for resolving trained policy checkpoints.

The helpers are used by both the RLBot wrapper and the Vutrium entrypoint so
the bot can find checkpoints consistently whether it is injected via DLL or run
through RLBot.
"""
from __future__ import annotations

import os
from pathlib import Path


def resolve_checkpoint_path(anchor: Path) -> str:
    """Resolve the checkpoint path or directory to load.

    Resolution order:
    1) ``AERIAL_BOT_CHECKPOINT`` environment variable (file or directory).
    2) ``checkpoints`` folder next to the provided anchor file.
    3) A clear prompt describing how to configure checkpoints.
    """

    env_path = os.getenv("AERIAL_BOT_CHECKPOINT")
    if env_path:
        resolved = Path(env_path).expanduser()
        if resolved.exists():
            return str(resolved)
        print(
            "AERIAL_BOT_CHECKPOINT is set but does not exist: "
            f"{resolved}. Falling back to ./checkpoints."
        )

    default_dir = anchor.parent / "checkpoints"
    default_dir = default_dir.expanduser()
    if default_dir.exists():
        return str(default_dir.resolve())

    print(
        "No checkpoint configured. Set AERIAL_BOT_CHECKPOINT to a policy file "
        "or directory, or place .pt/.pth files in ./checkpoints next to "
        f"{anchor.name}."
    )
    return str(default_dir.resolve())

