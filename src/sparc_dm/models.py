"""Dark-matter halo density/velocity models and baryonic rotation-curve physics."""

from __future__ import annotations

import numpy as np

from .data import GalaxyData
from .guards import check_chi2_inputs, check_halo_inputs, safe_iso_mass_factor, safe_nfw_mass_factor

G_ASTRO: float = 4.30091e-6  # kpc * (km/s)^2 / M_sun
UPSILON_BUL: float = 0.7


def _mass_to_velocity(r_arr: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Compute circular rotation speed ``sqrt(G * M(r) / r)`` in km/s."""
    v_sq = np.zeros(r_arr.shape, dtype=np.float64)
    pos = r_arr > 0.0
    v_sq[pos] = G_ASTRO * mass[pos] / r_arr[pos]
    return np.sqrt(v_sq)


def iso_density(r: np.ndarray | float, rho0: float, rc: float) -> np.ndarray:
    """Pseudo-Isothermal core density ``rho(r) = rho0 / (1 + (r/rc)^2)``."""
    x = check_halo_inputs(rho0, rc, r, "ISO") / rc
    return rho0 / (1.0 + x * x)


def iso_mass_exact(r: np.ndarray | float, rho0: float, rc: float) -> np.ndarray:
    """Analytical enclosed mass ``M_ISO(r) = 4*pi*rho0*rc^3 * (x - arctan(x))``."""
    x = check_halo_inputs(rho0, rc, r, "ISO") / rc
    return 4.0 * np.pi * rho0 * (rc**3) * safe_iso_mass_factor(x)


def iso_velocity(r: np.ndarray | float, rho0: float, rc: float) -> np.ndarray:
    """Circular velocity ``sqrt(G * M_ISO(r) / r)`` in km/s."""
    r_arr = check_halo_inputs(rho0, rc, r, "ISO")
    mass = 4.0 * np.pi * rho0 * (rc**3) * safe_iso_mass_factor(r_arr / rc)
    return _mass_to_velocity(r_arr, mass)


def nfw_density(r: np.ndarray | float, rhos: float, rs: float) -> np.ndarray:
    """NFW cusp density ``rho(r) = rhos / ((r/rs) * (1 + r/rs)^2)``."""
    x = check_halo_inputs(rhos, rs, r, "NFW") / rs
    if rhos == 0.0:
        return np.zeros_like(x)
    pos = x > 0.0
    xp = np.where(pos, x, 1.0)
    return np.where(pos, rhos / (xp * (1.0 + xp) ** 2), np.inf)


def nfw_mass_exact(r: np.ndarray | float, rhos: float, rs: float) -> np.ndarray:
    """Analytical enclosed mass ``M_NFW(r) = 4*pi*rhos*rs^3 * (ln(1+x) - x/(1+x))``."""
    x = check_halo_inputs(rhos, rs, r, "NFW") / rs
    return 4.0 * np.pi * rhos * (rs**3) * safe_nfw_mass_factor(x)


def nfw_velocity(r: np.ndarray | float, rhos: float, rs: float) -> np.ndarray:
    """Circular velocity ``sqrt(G * M_NFW(r) / r)`` in km/s."""
    r_arr = check_halo_inputs(rhos, rs, r, "NFW")
    mass = 4.0 * np.pi * rhos * (rs**3) * safe_nfw_mass_factor(r_arr / rs)
    return _mass_to_velocity(r_arr, mass)


def baryonic_velocity_sq(galaxy: GalaxyData, upsilon_disk: float = 0.5) -> np.ndarray:
    """Signed baryonic circular velocity squared preserving negative inner gas terms."""
    return galaxy.v_gas * np.abs(galaxy.v_gas) + upsilon_disk * galaxy.v_disk**2 + UPSILON_BUL * galaxy.v_bul**2


def total_velocity(v_bar_sq: np.ndarray | float, v_dm: np.ndarray | float) -> np.ndarray:
    """Combine baryonic velocity squared and dark-matter halo velocity in quadrature."""
    return np.sqrt(np.maximum(np.asarray(v_bar_sq, dtype=np.float64) + np.asarray(v_dm, dtype=np.float64) ** 2, 0.0))


def chi2_reduced(v_obs: np.ndarray, v_err: np.ndarray, v_model: np.ndarray, n_params: int = 0) -> float:
    """Reduced chi-squared statistic with degrees of freedom ``max(N - n_params, 1)``."""
    obs, err, mod, n_p = check_chi2_inputs(v_obs, v_err, v_model, n_params)
    return float(np.sum(((obs - mod) / err) ** 2) / max(obs.size - n_p, 1))
