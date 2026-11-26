"""2D galactic test-particle ODE solvers (Euler, RK4, Velocity Verlet)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .guards import check_orbit_init, check_pos, check_traj, safe_iso_potential_term
from .models import G_ASTRO

KPC_PER_GYR_PER_KMS = 1.022712  # 1 km/s = 1.022712 kpc/Gyr
MassFunc = Callable[[float], float]
PotFunc = Callable[[float], float]


def accel(pos: np.ndarray, mass_fn: MassFunc) -> np.ndarray:
    """Radial gravitational acceleration ``-G * M(r) * pos / r^3`` in ``(km/s)^2 / kpc``."""
    p, r = check_pos(pos)
    return -(G_ASTRO * float(mass_fn(r)) / r) * (p / r) / r


def iso_potential(r: float, rho0: float, rc: float) -> float:
    """Analytical ISO gravitational potential ``Phi(r) = 4*pi*G*rho0*rc^2 * (0.5 ln(1+x^2) + arctan(x)/x)``."""
    return float(4.0 * np.pi * G_ASTRO * rho0 * (rc**2) * safe_iso_potential_term(r, rho0, rc))


def _init_traj(r0_kpc: float, v0_kms: float, dt_gyr: float, n_steps: int) -> tuple[np.ndarray, int]:
    steps = check_orbit_init(r0_kpc, v0_kms, dt_gyr, n_steps)
    traj = np.empty((steps + 1, 4), dtype=np.float64)
    traj[0] = (r0_kpc, 0.0, 0.0, v0_kms)
    return traj, steps


def _rhs(state: np.ndarray, mass_fn: MassFunc) -> np.ndarray:
    a = accel(state[:2], mass_fn)
    return np.array([state[2], state[3], a[0], a[1]], dtype=np.float64) * KPC_PER_GYR_PER_KMS


def euler_orbit(r0_kpc: float, v0_kms: float, mass_fn: MassFunc, dt_gyr: float, n_steps: int) -> np.ndarray:
    """Forward Euler ``O(dt)`` 2D orbit integration."""
    traj, steps = _init_traj(r0_kpc, v0_kms, dt_gyr, n_steps)
    for i in range(steps):
        traj[i + 1] = traj[i] + dt_gyr * _rhs(traj[i], mass_fn)
    return traj


def rk4_orbit(r0_kpc: float, v0_kms: float, mass_fn: MassFunc, dt_gyr: float, n_steps: int) -> np.ndarray:
    """Classical 4th-order Runge-Kutta ``O(dt^4)`` 2D orbit integration."""
    traj, steps = _init_traj(r0_kpc, v0_kms, dt_gyr, n_steps)
    for i in range(steps):
        y = traj[i]
        k1 = _rhs(y, mass_fn)
        k2 = _rhs(y + 0.5 * dt_gyr * k1, mass_fn)
        k3 = _rhs(y + 0.5 * dt_gyr * k2, mass_fn)
        k4 = _rhs(y + dt_gyr * k3, mass_fn)
        traj[i + 1] = y + (dt_gyr / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return traj


def velocity_verlet_orbit(r0_kpc: float, v0_kms: float, mass_fn: MassFunc, dt_gyr: float, n_steps: int) -> np.ndarray:
    """Symplectic Velocity Verlet (kick-drift-kick) 2D orbit integration."""
    traj, steps = _init_traj(r0_kpc, v0_kms, dt_gyr, n_steps)
    c, pos, vel = KPC_PER_GYR_PER_KMS, traj[0, :2].copy(), traj[0, 2:].copy()
    acc = accel(pos, mass_fn) * c
    for i in range(steps):
        vel_half = vel + 0.5 * dt_gyr * acc
        pos = pos + dt_gyr * (vel_half * c)
        acc = accel(pos, mass_fn) * c
        vel = vel_half + 0.5 * dt_gyr * acc
        traj[i + 1, :2], traj[i + 1, 2:] = pos, vel
    return traj


def orbit_diagnostics(traj: np.ndarray, pot_fn: PotFunc) -> dict[str, float]:
    """Return radial ratios, gauge-invariant energy drift (abs and /K0), and angular momentum drift."""
    arr, r = check_traj(traj)
    kin = 0.5 * (arr[:, 2] ** 2 + arr[:, 3] ** 2)
    energy = kin + np.array([float(pot_fn(float(ri))) for ri in r])
    ang_mom = arr[:, 0] * arr[:, 3] - arr[:, 1] * arr[:, 2]
    max_abs_e = float(np.max(np.abs(energy - energy[0])))
    return {
        "r_max_ratio": float(np.max(r) / r[0]),
        "r_final_ratio": float(r[-1] / r[0]),
        "abs_E_err": max_abs_e,
        "rel_K_err": float(max_abs_e / max(float(kin[0]), 1e-12)),
        "rel_E_err": float(max_abs_e / max(abs(float(energy[0])), 1e-12)),
        "rel_L_err": float(np.max(np.abs(ang_mom - ang_mom[0])) / max(abs(float(ang_mom[0])), 1e-12)),
    }
