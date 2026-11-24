"""SPARC galaxy rotation-curve parser and catalog loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from .guards import check_galaxy_table, require_int

_DISTANCE_RE = re.compile(r"Distance\s*=\s*([0-9.eE+-]+)")


@dataclass(frozen=True)
class GalaxyData:
    """Observed rotation curve and baryonic velocity components for one galaxy."""

    name: str
    distance_mpc: float
    rad_kpc: np.ndarray
    v_obs: np.ndarray
    v_err: np.ndarray
    v_gas: np.ndarray
    v_disk: np.ndarray
    v_bul: np.ndarray


def load_galaxy(path: str | Path) -> GalaxyData:
    """Load and parse a single SPARC ``*_rotmod.dat`` file."""
    file_path = Path(path)
    distance_mpc: float | None = None
    rows: list[list[float]] = []

    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                m = _DISTANCE_RE.search(s)
                if m:
                    distance_mpc = float(m.group(1))
                continue
            parts = s.split()
            if len(parts) < 6:
                raise ValueError(f"Expected >= 6 columns in {file_path.name}: {s!r}")
            rows.append([float(parts[i]) for i in range(6)])

    t = np.asarray(rows, dtype=np.float64)
    dist = check_galaxy_table(t, distance_mpc, file_path.name)
    return GalaxyData(
        name=file_path.stem.removesuffix("_rotmod"),
        distance_mpc=dist,
        rad_kpc=t[:, 0].copy(),
        v_obs=t[:, 1].copy(),
        v_err=t[:, 2].copy(),
        v_gas=t[:, 3].copy(),
        v_disk=t[:, 4].copy(),
        v_bul=t[:, 5].copy(),
    )


def load_sparc_catalog(data_dir: str | Path, min_points: int = 5) -> dict[str, GalaxyData]:
    """Load all SPARC ``*_rotmod.dat`` galaxies with at least ``min_points`` radii."""
    n_min = require_int(min_points, 1, "min_points")
    root = Path(data_dir)
    files = sorted(root.glob("*_rotmod.dat")) if root.is_dir() else []
    if not files:
        raise FileNotFoundError(f"No *_rotmod.dat files found in {root}")
    return {g.name: g for fp in files if (g := load_galaxy(fp)).rad_kpc.size >= n_min}
