"""Input validation, small-r Taylor asymptotics, overflow guards, and box-bound helpers."""

from __future__ import annotations

import numpy as np


def require_int(value: int, minimum: int, label: str) -> int:
    """Validate that ``value`` is a non-boolean integer ``>= minimum``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def check_galaxy_table(table: np.ndarray, distance_mpc: float | None, filename: str) -> float:
    """Validate SPARC galaxy distance header and strictly increasing positive radii/errors."""
    if distance_mpc is None or not np.isfinite(distance_mpc) or distance_mpc <= 0.0:
        raise ValueError(f"Invalid or missing distance header in {filename}")
    if (
        table.size == 0
        or not np.all(np.isfinite(table))
        or np.any(table[:, 0] <= 0.0)
        or np.any(np.diff(table[:, 0]) <= 0.0)
        or np.any(table[:, 1] < 0.0)
        or np.any(table[:, 2] <= 0.0)
    ):
        raise ValueError(f"Invalid radii, velocities, uncertainties, or non-finite values in {filename}")
    return float(distance_mpc)


def check_1d_grid(
    y: np.ndarray, min_size: int, h: float, label: str, require_even_intervals: bool = False
) -> np.ndarray:
    """Validate uniform-grid samples and finite spacing ``h`` for composite quadrature."""
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 1 or arr.size < min_size or not np.isfinite(h) or (require_even_intervals and (arr.size - 1) % 2 != 0):
        raise ValueError(f"Invalid inputs to {label}")
    return arr


def check_radial_inputs(r_grid: np.ndarray, method: str, n_nodes: int) -> np.ndarray:
    """Validate cumulative radial integration grid, quadrature method, and node count."""
    radii = np.asarray(r_grid, dtype=np.float64)
    if radii.ndim != 1 or np.any(radii < 0.0) or not np.all(np.isfinite(radii)):
        raise ValueError("r_grid must be a 1D array of finite non-negative radii")
    if method not in {"gauss", "simpson"}:
        raise ValueError(f"Unsupported quadrature method: {method!r}")
    if method == "simpson" and (require_int(n_nodes, 2, "n_nodes") % 2 != 0):
        raise ValueError("Simpson panel integration requires an even integer n_nodes >= 2")
    return radii


def check_halo_inputs(rho: float, r_scale: float, r: np.ndarray | float, label: str) -> np.ndarray:
    """Validate halo parameters (``rho >= 0``, ``r_scale > 0``) and finite non-negative radii."""
    if not (np.isfinite(rho) and np.isfinite(r_scale)) or rho < 0.0 or r_scale <= 0.0:
        raise ValueError(f"{label} parameters require finite density >= 0 and scale radius > 0")
    arr = np.asarray(r, dtype=np.float64)
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("Radii must be finite and non-negative")
    return arr


def safe_iso_mass_factor(x: np.ndarray) -> np.ndarray:
    """Evaluate ``x - arctan(x)`` using a 4-term Taylor series for ``x < 0.02``."""
    small = x < 0.02
    xs = np.where(small, x, 0.0)
    x2 = xs * xs
    taylor = (xs * x2) * (1.0 / 3.0 - x2 / 5.0 + (x2 * x2) / 7.0 - (x2 * x2 * x2) / 9.0)
    return np.where(small, taylor, x - np.arctan(x))


def safe_nfw_mass_factor(x: np.ndarray) -> np.ndarray:
    """Evaluate ``ln(1+x) - x/(1+x)`` using a 4-term Taylor series for ``x < 1e-3``."""
    small = x < 1e-3
    xs = np.where(small, x, 0.0)
    taylor = (xs * xs) * (0.5 - (2.0 / 3.0) * xs + 0.75 * xs * xs - 0.8 * (xs**3))
    return np.where(small, taylor, np.log1p(x) - x / (1.0 + x))


def check_chi2_inputs(
    v_obs: np.ndarray, v_err: np.ndarray, v_model: np.ndarray, n_params: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Validate shapes, finiteness, positive uncertainties, and free parameter count."""
    obs, err, mod = (np.asarray(a, dtype=np.float64) for a in (v_obs, v_err, v_model))
    n_p = require_int(n_params, 0, "n_params")
    if (
        obs.size == 0
        or obs.shape != err.shape
        or obs.shape != mod.shape
        or not (np.all(np.isfinite(obs)) and np.all(np.isfinite(err)) and np.all(np.isfinite(mod)))
        or np.any(err <= 0.0)
    ):
        raise ValueError("Invalid inputs to chi2_reduced")
    return obs, err, mod, n_p


def check_pos(pos: np.ndarray) -> tuple[np.ndarray, float]:
    """Validate finite 2D position vector and strictly positive radius."""
    p = np.asarray(pos, dtype=np.float64)
    if p.shape != (2,) or not np.all(np.isfinite(p)):
        raise ValueError("Position must be a finite 2D vector of shape (2,)")
    r = float(np.hypot(p[0], p[1]))
    if not np.isfinite(r) or r <= 0.0:
        raise ValueError("Position radius must be finite and strictly positive")
    return p, r


def safe_iso_potential_term(r: float, rho0: float, rc: float) -> float:
    """Validate ISO potential inputs and avoid cancellation/overflow at small or huge ``r/rc``."""
    if not (np.isfinite(r) and np.isfinite(rho0) and np.isfinite(rc)) or rho0 < 0.0 or rc <= 0.0 or r < 0.0:
        raise ValueError("iso_potential requires finite rho0 >= 0, rc > 0, and r >= 0")
    x = float(r) / rc
    if x < 1e-3:
        x2 = x * x
        return 1.0 + x2 / 6.0 - (x2 * x2) / 20.0
    if x > 1e8:
        return float(np.log(x) + np.arctan(x) / x)
    return float(0.5 * np.log1p(x * x) + np.arctan(x) / x)


def check_orbit_init(r0_kpc: float, v0_kms: float, dt_gyr: float, n_steps: int) -> int:
    """Validate initial orbit state and step controls."""
    steps = require_int(n_steps, 1, "n_steps")
    if not (np.isfinite(r0_kpc) and np.isfinite(v0_kms) and np.isfinite(dt_gyr)) or r0_kpc <= 0.0 or dt_gyr <= 0.0:
        raise ValueError("Expected finite r0_kpc > 0, v0_kms, dt_gyr > 0, and integer n_steps >= 1")
    return steps


def check_traj(traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate finite ``(N >= 1, 4)`` orbit trajectory and positive initial radius."""
    arr = np.asarray(traj, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != 4 or not np.all(np.isfinite(arr)):
        raise ValueError("traj must be a finite 2D array of shape (N >= 1, 4)")
    r = np.hypot(arr[:, 0], arr[:, 1])
    if r[0] <= 0.0:
        raise ValueError("Initial orbit radius r[0] must be strictly positive")
    return arr, r


def check_tol_iter(tol: float, max_iter: int) -> int:
    """Validate positive finite tolerance and integer iteration cap."""
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"Expected finite tol > 0, got {tol}")
    return require_int(max_iter, 1, "max_iter")


def check_interval(a: float, b: float, tol: float, max_iter: int) -> tuple[float, float, int]:
    """Validate finite ordered bracket ``a < b`` alongside tolerance and iteration cap."""
    n_max = check_tol_iter(tol, max_iter)
    if not (np.isfinite(a) and np.isfinite(b)) or a >= b:
        raise ValueError(f"Expected finite a < b, got [{a}, {b}]")
    return float(a), float(b), n_max


def check_newton_bounds(x0: float, bounds: tuple[float, float] | None) -> tuple[float, float, bool]:
    """Validate finite ``x0`` and optional bracketing interval ``[lo, hi]`` containing ``x0``."""
    if not np.isfinite(x0):
        raise ValueError(f"Expected finite x0, got {x0}")
    if bounds is None:
        return 0.0, 0.0, False
    b_lo, b_hi = float(bounds[0]), float(bounds[1])
    if not (np.isfinite(b_lo) and np.isfinite(b_hi)) or b_lo >= b_hi or not (b_lo <= x0 <= b_hi):
        raise ValueError(f"Expected finite bounds with lo <= x0 <= hi, got {bounds}, x0={x0}")
    return b_lo, b_hi, True


BoundsPair = tuple[np.ndarray, np.ndarray] | None


def check_cg_inputs(
    x0, tol: float, max_iter: int, max_step: float, bounds: BoundsPair
) -> tuple[np.ndarray, int, BoundsPair]:
    """Validate CG initial vector, step limit, and optional finite box bounds ``lo < hi``."""
    n_max = check_tol_iter(tol, max_iter)
    if not np.isfinite(max_step) or max_step <= 0.0:
        raise ValueError(f"Expected finite max_step > 0, got {max_step}")
    x = np.atleast_1d(np.asarray(x0, dtype=np.float64)).copy()
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("x0 must be a non-empty 1D array of finite numbers")
    if bounds is None:
        return x, n_max, None
    lo, hi = np.asarray(bounds[0], dtype=np.float64), np.asarray(bounds[1], dtype=np.float64)
    if lo.shape != x.shape or hi.shape != x.shape or not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))) or np.any(lo >= hi):
        raise ValueError("bounds must be finite (lo, hi) arrays matching x0 with lo < hi")
    return np.clip(x, lo, hi), n_max, (lo, hi)


def project_box_vector(x: np.ndarray, v: np.ndarray, bounds: BoundsPair, flip: float = 1.0) -> np.ndarray:
    """Zero components of gradient (``flip=1``) or direction (``flip=-1``) pushing out of active bounds."""
    if bounds is None:
        return v
    lo, hi = bounds
    out = v.copy()
    out[((x <= lo + 1e-12) & (flip * v > 0.0)) | ((x >= hi - 1e-12) & (flip * v < 0.0))] = 0.0
    return out


def active_set_changed(x_old: np.ndarray, x_new: np.ndarray, bounds: BoundsPair) -> bool:
    """Return True when iterates enter or leave box-constraint boundaries."""
    if bounds is None:
        return False
    lo, hi = bounds
    return bool(np.any(((x_old <= lo + 1e-12) | (x_old >= hi - 1e-12)) != ((x_new <= lo + 1e-12) | (x_new >= hi - 1e-12))))


def hit_bounds_labels(x: np.ndarray, bounds: BoundsPair, labels: tuple[str, ...], tol: float = 1e-4) -> list[str]:
    """Return bound-hit tags like ``upsilon_d_lo`` or ``r_scale_hi`` for coordinates within ``tol``."""
    if bounds is None:
        return []
    lo, hi = bounds
    arr = np.asarray(x, dtype=np.float64)
    hits: list[str] = []
    for i, name in enumerate(labels):
        if arr[i] <= lo[i] + tol:
            hits.append(f"{name}_lo")
        elif arr[i] >= hi[i] - tol:
            hits.append(f"{name}_hi")
    return hits
