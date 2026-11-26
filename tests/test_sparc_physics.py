"""SPARC catalog, rotation-curve fit, orbit, and CLI integration tests."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pytest

from sparc_dm.cli import baryon_mass_and_potential, run_survey
from sparc_dm.data import GalaxyData, load_galaxy, load_sparc_catalog
from sparc_dm.models import G_ASTRO, baryonic_velocity_sq, chi2_reduced, iso_mass_exact, iso_velocity, nfw_velocity, total_velocity
from sparc_dm.orbits import euler_orbit, iso_potential, orbit_diagnostics, rk4_orbit, velocity_verlet_orbit
from sparc_dm.solvers import baryonic_only_fit, find_crossover_radius, fit_galaxy_halo, halo_model_velocity

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SRC_DIR = ROOT / "src" / "sparc_dm"


def test_sparc_catalog_bulge_gas_and_parser_boundaries(tmp_path: Path) -> None:
    """Verify exact 171-galaxy min_points=5 catalog, signed gas, bulge scaling, and parser rejections."""
    catalog = load_sparc_catalog(DATA_DIR, min_points=5)
    assert len(catalog) == 171 and all(g.rad_kpc.size >= 5 for g in catalog.values())
    assert "D512-2" not in catalog

    ngc = catalog["NGC3198"]
    neg_mask = ngc.v_gas < 0.0
    assert np.isclose(ngc.distance_mpc, 13.8, rtol=1e-6) and np.any(neg_mask)
    assert np.allclose(baryonic_velocity_sq(ngc, upsilon_disk=0.0)[neg_mask], -(ngc.v_gas[neg_mask] ** 2))

    # Real bulge galaxy NGC2841 includes nonzero UPSILON_BUL * v_bul^2
    bul_gal = catalog["NGC2841"]
    assert np.any(bul_gal.v_bul > 0.0)
    diff_bul = baryonic_velocity_sq(bul_gal, 0.5) - baryonic_velocity_sq(replace(bul_gal, v_bul=np.zeros_like(bul_gal.v_bul)), 0.5)
    assert np.allclose(diff_bul, 0.7 * (bul_gal.v_bul**2))

    bad_files = {
        "nodist_rotmod.dat": "1.0 50.0 2.0 10.0 40.0 0.0\n",
        "short_rotmod.dat": "# Distance = 10.0 Mpc\n1.0 50.0 2.0\n",
        "unsorted_rotmod.dat": "# Distance = 10.0 Mpc\n2.0 50.0 2.0 0 40 0\n1.0 60.0 2.0 0 45 0\n",
        "zero_err_rotmod.dat": "# Distance = 10.0 Mpc\n1.0 50.0 0.0 0 40 0\n",
    }
    for fname, body in bad_files.items():
        p = tmp_path / fname
        p.write_text(body, encoding="utf-8")
        with pytest.raises(ValueError):
            load_galaxy(p)
    with pytest.raises(FileNotFoundError):
        load_sparc_catalog(tmp_path / "missing_dir")


def test_baryonic_failure_dark_halo_fits_and_nfw_trap() -> None:
    """Verify baryon-only recovery, dark-halo fits on NGC3198/DDO154, and NFW scale-radius convergence."""
    ngc = load_galaxy(DATA_DIR / "NGC3198_rotmod.dat")
    ddo = load_galaxy(DATA_DIR / "DDO154_rotmod.dat")

    v_syn_bar = total_velocity(baryonic_velocity_sq(ngc, upsilon_disk=0.85), 0.0)
    bar_rec = baryonic_only_fit(replace(ngc, v_obs=v_syn_bar))
    assert bar_rec.opt.converged and abs(bar_rec.upsilon_disk - 0.85) < 1e-4
    assert bar_rec.chi2_red < 1e-6 and bar_rec.rho0 == 0.0 and math.isnan(bar_rec.r_scale)

    ngc_bar, ngc_iso, ngc_nfw = baryonic_only_fit(ngc), fit_galaxy_halo(ngc, "iso"), fit_galaxy_halo(ngc, "nfw")
    assert ngc_bar.opt.converged and ngc_iso.opt.converged and ngc_nfw.opt.converged
    for fixed_u in (0.5, 2.0):
        assert ngc_bar.chi2_red < chi2_reduced(ngc.v_obs, ngc.v_err, total_velocity(baryonic_velocity_sq(ngc, fixed_u), 0.0), 1)
    assert ngc_bar.chi2_red > 100.0 and ngc_iso.chi2_red < 3.0 and ngc_nfw.chi2_red < 3.0

    ddo_bar, ddo_iso, ddo_nfw = baryonic_only_fit(ddo), fit_galaxy_halo(ddo, "iso"), fit_galaxy_halo(ddo, "nfw")
    assert ddo_bar.opt.converged and ddo_iso.opt.converged and ddo_nfw.opt.converged
    assert ddo_bar.chi2_red > 1000.0 and ddo_iso.chi2_red < 3.0 and ddo_iso.chi2_red < 0.5 * ddo_nfw.chi2_red

    for name, upper in (("NGC4389", 5.0), ("UGC07323", 1.0), ("UGC07577", 0.25)):
        fit_nfw = fit_galaxy_halo(load_galaxy(DATA_DIR / f"{name}_rotmod.dat"), "nfw")
        assert fit_nfw.opt.converged and fit_nfw.chi2_red < upper

    # Verify compact-core & outer-scale multi-start ISO fits avoid suboptimal basins
    fit_ngc2998 = fit_galaxy_halo(load_galaxy(DATA_DIR / "NGC2998_rotmod.dat"), "iso")
    assert fit_ngc2998.opt.converged and fit_ngc2998.chi2_red < 1.1
    fit_ugc7577 = fit_galaxy_halo(load_galaxy(DATA_DIR / "UGC07577_rotmod.dat"), "iso")
    assert fit_ugc7577.opt.converged and fit_ugc7577.chi2_red < 0.14

    # Verify multi-valley spirals converge cleanly for both halo profiles
    for name in ("NGC5055", "NGC5985", "NGC7814", "UGC03205"):
        gal_v = load_galaxy(DATA_DIR / f"{name}_rotmod.dat")
        for k in ("iso", "nfw"):
            assert fit_galaxy_halo(gal_v, k).opt.converged

    with pytest.raises(ValueError):
        halo_model_velocity(ngc, "invalid", 1e7, 2.0, 0.5)


def test_crossover_ngc3198_ddo154_and_safeguarded_newton() -> None:
    """Verify NGC3198 ISO/NFW crossovers, safeguarded Newton on F561-1/UGC11820, and non-crossing guards."""
    ngc = load_galaxy(DATA_DIR / "NGC3198_rotmod.dat")
    for kind, v_fn in (("iso", iso_velocity), ("nfw", nfw_velocity)):
        fit = fit_galaxy_halo(ngc, kind)
        bis, newt = find_crossover_radius(ngc, fit.rho0, fit.r_scale, fit.upsilon_disk, kind)
        assert bis.converged and newt.converged and abs(bis.root - newt.root) < 1e-6
        assert newt.iterations < bis.iterations
        vb2_at = float(np.interp(bis.root, ngc.rad_kpc, baryonic_velocity_sq(ngc, fit.upsilon_disk)))
        assert abs(float(v_fn(bis.root, fit.rho0, fit.r_scale) ** 2) - vb2_at) < 1e-3

    # Bracket-safeguarded Newton prevents boundary escape (F561-1) and false roots (UGC11820)
    b1, n1 = find_crossover_radius(load_galaxy(DATA_DIR / "F561-1_rotmod.dat"), 1e7, 2.0, 0.5, "iso")
    assert b1.converged and n1.converged and abs(b1.root - n1.root) < 1e-6
    b2, n2 = find_crossover_radius(load_galaxy(DATA_DIR / "UGC11820_rotmod.dat"), 1e6, 10.0, 0.5, "nfw")
    assert b2.converged and n2.converged and abs(b2.root - n2.root) < 1e-6

    ddo = load_galaxy(DATA_DIR / "DDO154_rotmod.dat")
    ddo_iso = fit_galaxy_halo(ddo, "iso")
    with pytest.raises(ValueError):
        find_crossover_radius(ddo, ddo_iso.rho0, ddo_iso.r_scale, ddo_iso.upsilon_disk, "iso")
    with pytest.raises(ValueError):
        find_crossover_radius(ngc, 0.0, 2.0, 0.5, "iso")

    # Outermost upward crossing selection on multi-crossing synthetic curve
    r_m = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    v_dm_m = iso_velocity(r_m, 5e7, 2.0)
    v_bar_m = np.array([v_dm_m[0] + 10.0, v_dm_m[1] - 5.0, v_dm_m[2] + 5.0, v_dm_m[3] - 10.0, v_dm_m[4] - 20.0])
    syn_multi = GalaxyData("MULTI", 10.0, r_m, v_dm_m, np.ones_like(r_m), np.zeros_like(r_m), v_bar_m, np.zeros_like(r_m))
    bm, nm = find_crossover_radius(syn_multi, 5e7, 2.0, 1.0, "iso")
    assert 3.0 < bm.root < 4.0 and abs(bm.root - nm.root) < 1e-6


def test_ngc3198_orbits_and_baryon_potential_consistency() -> None:
    """Verify bound dark-halo orbit, baryon-only escape, repulsive ejection, and dPhi/dr == G*M/r^2."""
    ngc = load_galaxy(DATA_DIR / "NGC3198_rotmod.dat")
    ngc_iso = fit_galaxy_halo(ngc, "iso")
    r0, dt, n_steps = 25.0, 0.01, 200
    v0 = float(np.interp(r0, ngc.rad_kpc, ngc.v_obs))

    m_bar, p_bar = baryon_mass_and_potential(ngc, ngc_iso.upsilon_disk)
    for r_chk in (0.5 * float(ngc.rad_kpc[0]), 12.0, float(ngc.rad_kpc[-1]) + 15.0):
        h = 1e-4 * r_chk
        assert np.isclose((p_bar(r_chk + h) - p_bar(r_chk - h)) / (2.0 * h), G_ASTRO * m_bar(r_chk) / (r_chk**2), rtol=2e-3)

    m_tot = lambda r: m_bar(r) + float(iso_mass_exact(r, ngc_iso.rho0, ngc_iso.r_scale))
    p_tot = lambda r: p_bar(r) + iso_potential(r, ngc_iso.rho0, ngc_iso.r_scale)

    diag_dm_rk4 = orbit_diagnostics(rk4_orbit(r0, v0, m_tot, dt, n_steps), p_tot)
    diag_dm_verlet = orbit_diagnostics(velocity_verlet_orbit(r0, v0, m_tot, dt, n_steps), p_tot)
    diag_dm_euler = orbit_diagnostics(euler_orbit(r0, v0, m_tot, dt, n_steps), p_tot)
    diag_bar_rk4 = orbit_diagnostics(rk4_orbit(r0, v0, m_bar, dt, n_steps), p_bar)
    diag_rep = orbit_diagnostics(rk4_orbit(r0, v0, lambda r: -m_tot(r), dt, n_steps), lambda r: -p_tot(r))

    assert diag_dm_rk4["r_max_ratio"] < 1.2 and 0.85 < diag_dm_rk4["r_final_ratio"] < 1.15 and diag_dm_rk4["rel_K_err"] < 1e-4
    assert diag_dm_verlet["r_max_ratio"] < 1.2 and diag_dm_verlet["rel_K_err"] < 1e-4 and diag_dm_verlet["rel_L_err"] < 1e-13
    assert diag_dm_euler["r_max_ratio"] > 1.3 and diag_dm_euler["rel_K_err"] > 0.5
    assert diag_bar_rk4["r_final_ratio"] > 1.8 and diag_bar_rk4["rel_K_err"] < 1e-4
    assert diag_rep["r_final_ratio"] > 15.0 and diag_rep["r_final_ratio"] > 2.0 * diag_bar_rk4["r_final_ratio"]


def test_live_survey_and_source_isolation(tmp_path: Path) -> None:
    """Run run_survey() on a SPARC subset and verify src/sparc_dm/ imports only stdlib and NumPy."""
    sub_dir = tmp_path / "subset"
    sub_dir.mkdir()
    for name in ("NGC3198", "DDO154", "NGC2841", "D512-2"):
        shutil.copy(DATA_DIR / f"{name}_rotmod.dat", sub_dir / f"{name}_rotmod.dat")
    out_json = tmp_path / "sub_results.json"

    run_survey(sub_dir, out_json)
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["survey"]["n_galaxies"] == 3
    assert data["survey"]["non_converged_iso"] == 0 and data["survey"]["non_converged_nfw"] == 0 and data["survey"]["non_converged_baryons"] == 0
    assert set(data["survey"]["iso_bounds_hits"]) == {"rho0", "rc", "upsilon_d"}
    assert set(data["survey"]["nfw_bounds_hits"]) == {"rhos", "rs", "upsilon_d"}
    assert data["survey"]["median_chi2_baryons"] > 10.0 * data["survey"]["median_chi2_iso"]
    assert data["deep_dives"]["NGC3198"]["crossover"]["dm_dominated_everywhere"] is False
    assert 5.0 < data["deep_dives"]["NGC3198"]["crossover"]["bisection_kpc"] < 9.0
    assert data["deep_dives"]["DDO154"]["crossover"]["dm_dominated_everywhere"] is True
    assert data["orbit_test"]["baryons_only_rk4"]["r_final_ratio"] > 5.0

    banned = {"scipy", "matplotlib", "pandas"}
    for py_file in sorted(SRC_DIR.glob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in banned, f"Banned import {alias.name!r} in {py_file.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in banned, f"Banned import {node.module!r} in {py_file.name}"
