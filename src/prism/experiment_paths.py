# -*- coding: utf-8 -*-
"""Timestamped experiment output directories."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

DEFAULT_EXPERIMENTS_ROOT = "exp"
DEFAULT_EXPERIMENTS_DATE_FORMAT = "%y_%m_%d"
DEFAULT_SCRATCH_DIRNAME_PREFIX = "exp"


def make_experiment_dir(
    experiment_name: str,
    *,
    experiments_root: str | Path = DEFAULT_EXPERIMENTS_ROOT,
    date_format: str = DEFAULT_EXPERIMENTS_DATE_FORMAT,
    dirname_prefix: str = DEFAULT_SCRATCH_DIRNAME_PREFIX,
    now: datetime | None = None,
) -> Path:
    """Create and return ``<root>/<date>/<timestamp>_<prefix>_<name>/``."""
    run_time = now or datetime.now()
    date_str = run_time.strftime(date_format)
    timestamp_str = run_time.strftime("%y_%m_%d_%H%M%S")
    safe_name = experiment_name.replace(" ", "_")
    run_leaf = f"{timestamp_str}_{dirname_prefix}_{safe_name}"
    output_dir = Path(experiments_root) / date_str / run_leaf
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_experiment_output_dir(
    output: Path | None,
    experiment_name: str,
    *,
    repo_root: Path | None = None,
    experiments_root: str | Path = DEFAULT_EXPERIMENTS_ROOT,
    date_format: str = DEFAULT_EXPERIMENTS_DATE_FORMAT,
    dirname_prefix: str = DEFAULT_SCRATCH_DIRNAME_PREFIX,
    now: datetime | None = None,
) -> Path:
    """Use an explicit output path or create a timestamped experiment directory."""
    if output is not None:
        return Path(output)
    root = Path(experiments_root)
    if not root.is_absolute() and repo_root is not None:
        root = repo_root / root
    return make_experiment_dir(
        experiment_name,
        experiments_root=root,
        date_format=date_format,
        dirname_prefix=dirname_prefix,
        now=now,
    )
