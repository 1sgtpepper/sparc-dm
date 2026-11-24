"""Numerical method tests for sparc_dm."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import warnings

import numpy as np
import pytest

from sparc_dm.data import load_galaxy
from sparc_dm.integrators import (
    gauss_legendre,
    integrate_radial_mass_grid,
    simpson_integrate,
    trapezoid_integrate,
)
from sparc_dm.models import (
    G_ASTRO,
    chi2_reduced,
    iso_density,
    iso_mass_exact,
    iso_velocity,
    nfw_density,
    nfw_mass_exact,
    nfw_velocity,
    total_velocity,
)
from sparc_dm.orbits import (
    KPC_PER_GYR_PER_KMS,
    accel,
    euler_orbit,
    iso_potential,
    orbit_diagnostics,
    rk4_orbit,
    velocity_verlet_orbit,
)
from sparc_dm.solvers import bisection, brent_minimize, conjugate_gradient, fit_galaxy_halo, newton_raphson

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Independent SI physical reference constants (CODATA 2018 / IAU 2012-2015)
_G_SI = 6.67430e-11
_M_SUN_KG = 1.98847e30
_KPC_M = 3.0856775814913673e19
_GYR_S = 365.25 * 86400.0 * 1.0e9


def test_si_constants_halo_velocities_and_loss_validation() -> None:
    """Verify astrophysical constants against SI, ISO/NFW velocities against quadrature, and bounds."""
    g_si_ref = _G_SI * _M_SUN_KG / (_KPC_M * 1.0e6)
    c_si_ref = 1.0e3 * _GYR_S / _KPC_M
    assert np.isclose(G_ASTRO, g_si_ref, rtol=1e-4)
    assert np.isclose(KPC_PER_GYR_PER_KMS, c_si_ref, rtol=1e-5)

    # Compare analytical velocities against 16-pt Gauss-Legendre shell integration
    r_chk = np.array([1e-3, 1.0, 5.0])
    rho0, rc = 5.0e7, 2.5
    m_iso_quad = integrate_radial_mass_grid(lambda r: rho0 / (1.0 + (r / rc) ** 2), r_chk, "gauss", 16)
    assert np.allclose(iso_velocity(r_chk, rho0, rc), np.sqrt(g_si_ref * m_iso_quad / r_chk), rtol=1e-4)
    assert float(iso_velocity(0.0, rho0, rc)) == 0.0

    rhos, rs = 7.0e6, 12.0
    m_nfw_quad = integrate_radial_mass_grid(lambda r: rhos / ((r / rs) * (1.0 + r / rs) ** 2), r_chk, "gauss", 16)
    assert np.allclose(nfw_velocity(r_chk, rhos, rs), np.sqrt(g_si_ref * m_nfw_quad / r_chk), rtol=1e-4)
    assert float(nfw_velocity(0.0, rhos, rs)) == 0.0
    assert float(nfw_density(0.0, 0.0, rs)) == 0.0 and math.isinf(float(nfw_density(0.0, rhos, rs)))

    # Smoothness across small-x Taylor switch at x=0.02 without RuntimeWarnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        m_lo = float(iso_mass_exact(0.019999, 1.0 / (4.0 * math.pi), 1.0))
        m_hi = float(iso_mass_exact(0.020001, 1.0 / (4.0 * math.pi), 1.0))
        assert abs(m_hi / (0.020001**3) - m_lo / (0.019999**3)) < 1e-7
        assert np.isclose(iso_potential(0.0, rho0, rc), 4.0 * math.pi * G_ASTRO * rho0 * rc**2, rtol=1e-14)

    for bad_r in (-1.0, float("nan")):
        with pytest.raises(ValueError):
            iso_velocity(bad_r, rho0, rc)
        with pytest.raises(ValueError):
            nfw_velocity(bad_r, rhos, rs)

    assert float(total_velocity(-100.0, 5.0)) == 0.0
    v_obs, v_err, v_mod = np.array([10.0, 20.0, 30.0]), np.array([1.0, 2.0, 5.0]), np.array([11.0, 18.0, 25.0])
    assert np.isclose(chi2_reduced(v_obs, v_err, v_mod, n_params=0), 1.0)
    assert np.isclose(chi2_reduced(v_obs, v_err, v_mod, n_params=2), 3.0)
    assert np.isclose(chi2_reduced(v_obs, v_err, v_mod, n_params=5), 3.0)  # dof floor = 1
    with pytest.raises(ValueError):
        chi2_reduced(v_obs, np.array([1.0, 0.0, 1.0]), v_mod)
    with pytest.raises(ValueError):
        chi2_reduced(v_obs, np.array([1.0, float("nan"), 1.0]), v_mod)


@pytest.mark.parametrize(("n", "bounds"), [(1, (0.7, 3.9)), (2, (0.7, 3.9)), (3, (-1.0, 1.0)), (5, (0.7, 3.9))])
def test_gauss_legendre_moments_and_degree_2n_bound(n: int, bounds: tuple[float, float]) -> None:
    """n-point Gauss-Legendre integrates degree <= 2n-1 exactly and misses degree 2n predictably."""
    a, b = bounds
    nodes, weights = gauss_legendre(n, a, b)
    assert nodes.shape == (n,) and np.all(np.diff(nodes) > 0.0) and np.all((nodes > a) & (nodes < b))
    for deg in range(2 * n):
        exact = (b ** (deg + 1) - a ** (deg + 1)) / (deg + 1)
        assert np.isclose(float(np.dot(weights, nodes**deg)), exact, rtol=1e-12, atol=1e-13)

    if n == 3 and bounds == (-1.0, 1.0):
        assert np.isclose(float(np.dot(weights, nodes**6)), 6.0 / 25.0, rtol=1e-14)
        with pytest.raises(ValueError):
            gauss_legendre(0)


def test_trapezoid_simpson_orders_and_radial_mass_grid() -> None:
    """Verify Trapezoid O(h^2) vs Simpson O(h^4) convergence and radial mass grid against exact ISO/NFW."""
    x_lin = np.linspace(1.0, 4.0, 5)
    assert np.isclose(trapezoid_integrate(2.0 * x_lin + 3.0, 0.75), 24.0, rtol=1e-14)
    x_cub = np.linspace(1.0, 3.0, 3)
    assert np.isclose(simpson_integrate(x_cub**3 - 2.0 * x_cub + 1.0, 1.0), 14.0, rtol=1e-14)

    rho0, rc, r_max = 4.5e7, 2.3, 6.0
    m_exact = 4.0 * math.pi * rho0 * (rc**3) * (r_max / rc - math.atan(r_max / rc))
    hs, err_trap, err_simp = [], [], []
    for n_panels in (20, 40, 80, 160):
        r = np.linspace(0.0, r_max, n_panels + 1, dtype=np.float64)
        h = float(r[1] - r[0])
        integrand = 4.0 * math.pi * (r**2) * (rho0 / (1.0 + (r / rc) ** 2))
        hs.append(h)
        err_trap.append(abs(trapezoid_integrate(integrand, h) - m_exact))
        err_simp.append(abs(simpson_integrate(integrand, h) - m_exact))

    assert 1.95 < float(np.polyfit(np.log(hs), np.log(err_trap), 1)[0]) < 2.05
    assert 3.90 < float(np.polyfit(np.log(hs), np.log(err_simp), 1)[0]) < 4.10
    assert err_simp[-1] < 1e-4 * err_trap[-1]

    r_grid = np.array([4.5, 0.2, 18.0, 1.0])
    assert np.allclose(
        integrate_radial_mass_grid(lambda r: iso_density(r, 5e7, 2.5), r_grid, "gauss", 16),
        iso_mass_exact(r_grid, 5e7, 2.5),
        rtol=1e-12,
    )
    assert np.allclose(
        integrate_radial_mass_grid(lambda r: nfw_density(r, 7e6, 12.0), r_grid, "simpson", 32),
        nfw_mass_exact(r_grid, 7e6, 12.0),
        rtol=1e-7,
    )
    for bad_call in (
        lambda: simpson_integrate(np.array([1.0, 2.0, 3.0, 4.0]), 0.5),
        lambda: trapezoid_integrate(np.array([1.0]), 0.5),
        lambda: integrate_radial_mass_grid(lambda r: np.ones_like(r), np.array([-1.0, 1.0])),
        lambda: integrate_radial_mass_grid(lambda r: np.ones_like(r), np.array([1.0]), method="simpson", n_nodes=3),
    ):
        with pytest.raises(ValueError):
            bad_call()


def test_bisection_linear_vs_newton_quadratic_and_guards() -> None:
    """Verify increasing/decreasing bisection vs quadratic Newton contraction and failure boundaries."""
    f = lambda x: math.exp(x) - 3.0 * x - 2.0
    df = lambda x: math.exp(x) - 3.0

    bis = bisection(f, 1.0, 3.0, tol=1e-10)
    bis_dec = bisection(lambda x: -f(x), 1.0, 3.0, tol=1e-10)
    newt = newton_raphson(f, df, x0=2.8, tol=1e-12)

    assert bis.converged and bis_dec.converged and newt.converged
    assert bis.f_root == f(bis.root) and newt.f_root == f(newt.root)
    assert abs(bis.root - newt.root) < 1e-9 and abs(bis_dec.root - newt.root) < 1e-9
    assert 30 <= bis.iterations <= 35 and newt.iterations <= 7

    errs = [abs(x - newt.root) for x in newt.trace if abs(x - newt.root) > 1e-14]
    assert len(errs) >= 3 and errs[-1] < 2.0 * (errs[-2] ** 2)

    # Exact endpoints & non-convergence guards
    assert bisection(lambda x: x - 2.0, 2.0, 5.0).iterations == 0
    assert newton_raphson(lambda x: x * (2.0 - x), lambda x: 2.0 - 2.0 * x, x0=2.5, bounds=(2.0, 3.0)).root == 2.0
    assert newton_raphson(lambda x: x * x, lambda x: 2.0 * x, x0=0.0).converged
    assert not newton_raphson(lambda x: x * x - 1.0, lambda x: 2.0 * x, x0=0.0).converged
    assert not bisection(lambda x: x - 0.3, 0.0, 1.0, tol=1e-12, max_iter=2).converged

    for bad_call in (
        lambda: bisection(lambda x: x * x + 1.0, -1.0, 1.0),
        lambda: bisection(lambda x: x, 2.0, 1.0),
        lambda: bisection(lambda x: float("nan") if abs(x) < 0.1 else x, -1.0, 1.0),
        lambda: newton_raphson(lambda x: x, lambda x: 1.0, float("nan")),
    ):
        with pytest.raises(ValueError):
            bad_call()


def test_brent_cg_and_independent_halo_recovery() -> None:
    """Verify Brent minimization, conjugate gradient guards, and halo parameter recovery."""
    res = brent_minimize(lambda x: math.exp(x) - 4.0 * x, 0.0, 3.0, tol=1e-8)
    assert res.converged and res.iterations <= 15 and abs(res.x - math.log(4.0)) < 1e-7
    assert brent_minimize(lambda x: x, 0.1, 2.0, tol=1e-6).x < 0.10001
    assert not brent_minimize(lambda _x: float("nan"), 0.0, 1.0).converged
    with pytest.raises(ValueError):
        brent_minimize(lambda x: x * x, 2.0, 2.0)

    quad = lambda v: float(3.0 * (v[0] - 1.7) ** 2 + 2.0 * (v[0] - 1.7) * (v[1] + 0.8) + 7.0 * (v[1] + 0.8) ** 2)
    q_res = conjugate_gradient(quad, np.array([-4.0, 5.0]), tol=1e-7, max_iter=40)
    assert q_res.converged and q_res.iterations <= 4 and np.allclose(q_res.x, [1.7, -0.8], atol=1e-5)

    rosen = lambda v: float((1.0 - v[0]) ** 2 + 100.0 * (v[1] - v[0] ** 2) ** 2)
    r_res = conjugate_gradient(rosen, np.array([-1.2, 1.0]), tol=1e-6, max_iter=200)
    assert r_res.converged and np.allclose(r_res.x, [1.0, 1.0], atol=1e-3)

    # Non-finite ray evaluations and non-smooth walls must not report convergence
    nan_res = conjugate_gradient(lambda v: float("nan") if v[0] < 0.999 else float(v[0] ** 2), [1.0, 0.0], max_iter=5)
    assert not nan_res.converged and np.all(np.isfinite(nan_res.x)) and np.isfinite(nan_res.fun)
    wall = lambda v: float((v[0] - 10.0) ** 2 + 1e8 * max(0.0, -v[1]) ** 2 + 1e4 * max(0.0, v[1]))
    assert not conjugate_gradient(wall, [0.0, 0.0], max_iter=5).converged

    # Recover ISO and NFW parameters from quadrature-synthesized rotation curves
    ngc = load_galaxy(DATA_DIR / "NGC3198_rotmod.dat")
    r = ngc.rad_kpc
    for kind, true_rho, true_r, true_ups, offset in (
        ("iso", 6.0e7, 3.2, 0.55, True),
        ("nfw", 8.0e6, 14.0, 0.48, False),
    ):
        rho_fn = (
            (lambda rr: true_rho / (1.0 + (rr / true_r) ** 2))
            if kind == "iso"
            else (lambda rr: true_rho / ((rr / true_r) * (1.0 + rr / true_r) ** 2))
        )
        m_dm = integrate_radial_mass_grid(rho_fn, r, "gauss", 16)
        vb2 = ngc.v_gas * np.abs(ngc.v_gas) + true_ups * (ngc.v_disk**2) + 0.7 * (ngc.v_bul**2)
        v_syn = np.sqrt(np.maximum(vb2 + G_ASTRO * m_dm / r, 0.0))
        syn_gal = replace(ngc, v_obs=v_syn, v_err=np.full_like(v_syn, 2.0))
        x0 = np.log10([true_rho, true_r, true_ups]) + np.array([0.8, -0.5, -0.3]) if offset else None
        fit = fit_galaxy_halo(syn_gal, model_kind=kind, x0=x0)
        assert fit.opt.converged and fit.chi2_red < 1e-8
        assert abs(fit.rho0 - true_rho) / true_rho < 1e-3
        assert abs(fit.r_scale - true_r) / true_r < 1e-3
        assert abs(fit.upsilon_disk - true_ups) / true_ups < 1e-3


def test_kepler_orbit_convergence_orders_and_physics_controls() -> None:
    """Verify orbit solver convergence orders, conservation diagnostics, and potential gradients."""
    m_pt, r0 = 1.0e11, 10.0
    v_circ = math.sqrt(G_ASTRO * m_pt / r0)
    t_orb = 2.0 * math.pi * r0 / (v_circ * KPC_PER_GYR_PER_KMS)
    mass_fn = lambda _r: m_pt
    pot_fn = lambda r: -G_ASTRO * m_pt / r

    step_counts = (80, 160, 320, 640)
    dts = [t_orb / n for n in step_counts]
    err_verlet, err_rk4 = [], []
    for n, dt in zip(step_counts, dts):
        tr_v = velocity_verlet_orbit(r0, v_circ, mass_fn, dt, n)
        tr_r = rk4_orbit(r0, v_circ, mass_fn, dt, n)
        err_verlet.append(float(np.hypot(tr_v[-1, 0] - r0, tr_v[-1, 1])))
        err_rk4.append(float(np.hypot(tr_r[-1, 0] - r0, tr_r[-1, 1])))

    euler_counts = (640, 1280, 2560)
    dts_e = [t_orb / n for n in euler_counts]
    err_euler = []
    for n, dt in zip(euler_counts, dts_e):
        tr_e = euler_orbit(r0, v_circ, mass_fn, dt, n)
        err_euler.append(float(np.hypot(tr_e[-1, 0] - r0, tr_e[-1, 1])))
    assert 0.90 < float(np.polyfit(np.log(dts_e), np.log(err_euler), 1)[0]) < 1.10
    assert 1.95 < float(np.polyfit(np.log(dts), np.log(err_verlet), 1)[0]) < 2.05
    assert 3.85 < float(np.polyfit(np.log(dts), np.log(err_rk4), 1)[0]) < 4.25
    assert err_rk4[-1] < 1e-4 * err_verlet[-1]

    v_ecc = 1.2 * v_circ
    diag_v = orbit_diagnostics(velocity_verlet_orbit(r0, v_ecc, mass_fn, t_orb / 200.0, 1000), pot_fn)
    diag_e = orbit_diagnostics(euler_orbit(r0, v_ecc, mass_fn, t_orb / 200.0, 1000), pot_fn)
    assert diag_v["rel_E_err"] < 1e-3 and diag_v["rel_L_err"] < 1e-13 and diag_e["rel_E_err"] > 0.2

    # Repulsive mass trajectory and ISO potential gradient check
    radii_rep = np.hypot(*(rk4_orbit(r0, v_circ, lambda _r: -m_pt, t_orb / 100.0, 100)[:, :2].T))
    assert radii_rep[-1] / r0 > 5.0 and np.all(np.diff(radii_rep) > 0.0)

    rho0, rc, r_test, h = 5e7, 2.5, 7.3, 1e-5
    dphi_dr = (iso_potential(r_test + h, rho0, rc) - iso_potential(r_test - h, rho0, rc)) / (2.0 * h)
    assert np.isclose(dphi_dr, G_ASTRO * float(iso_mass_exact(r_test, rho0, rc)) / (r_test**2), rtol=1e-8)

    for bad_call in (
        lambda: accel(np.array([3.0, 4.0, 12.0]), mass_fn),
        lambda: accel(np.array([0.0, 0.0]), mass_fn),
        lambda: rk4_orbit(-1.0, v_circ, mass_fn, 0.01, 10),
        lambda: iso_potential(-1.0, rho0, rc),
        lambda: orbit_diagnostics(np.zeros((2, 3)), pot_fn),
    ):
        with pytest.raises(ValueError):
            bad_call()
