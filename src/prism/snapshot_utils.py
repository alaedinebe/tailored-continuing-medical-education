# -*- coding: utf-8 -*-
"""Numbered head-CSV snapshots for pipeline inspection."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class SnapshotSaver:
    """Save numbered ``DataFrame.head(max_rows)`` CSVs for quick inspection."""

    out_dir: str
    enabled: bool = True
    max_rows: int = 3
    _index: int = 1

    def save(self, df: pd.DataFrame, name: str) -> None:
        """Write ``{index:02d}_{name}_{rows}x{cols}.csv`` under ``out_dir``."""
        if not self.enabled or df is None:
            return
        if not isinstance(df, pd.DataFrame) or df.empty:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        shape = f"{df.shape[0]}x{df.shape[1]}"
        filename = os.path.join(self.out_dir, f"{self._index:02d}_{name}_{shape}.csv")
        df.head(self.max_rows).to_csv(filename, index=False)
        self._index += 1

    def save_manifest(self, df: pd.DataFrame, name: str, *, full: bool = False) -> None:
        """Write a metadata table; use ``full=True`` to skip the ``max_rows`` cap."""
        if not self.enabled or df is None:
            return
        if not isinstance(df, pd.DataFrame) or df.empty:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        shape = f"{df.shape[0]}x{df.shape[1]}"
        filename = os.path.join(self.out_dir, f"{self._index:02d}_{name}_{shape}.csv")
        out = df if full else df.head(self.max_rows)
        out.to_csv(filename, index=False)
        self._index += 1


def snapshot_settings_from_cfg(cfg: dict) -> tuple[bool, int]:
    """Return ``(enabled, max_rows)`` from a YAML config dict."""
    snap_cfg = cfg.get("snapshots", {}) or {}
    enabled = bool(snap_cfg.get("enabled", True))
    max_rows = int(snap_cfg.get("max_rows", 3))
    return enabled, max_rows


def make_snapshot_saver(
    out_dir: str,
    cfg: dict,
    *,
    start_index: int = 1,
) -> SnapshotSaver:
    """Build a ``SnapshotSaver`` using ``snapshots`` keys from *cfg*."""
    enabled, max_rows = snapshot_settings_from_cfg(cfg)
    saver = SnapshotSaver(out_dir=out_dir, enabled=enabled, max_rows=max_rows)
    saver._index = start_index
    return saver
