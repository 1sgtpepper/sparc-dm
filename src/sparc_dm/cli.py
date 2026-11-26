"""CLI entrypoint for the SPARC dark matter halo survey."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .data import GalaxyData, load_sparc_catalog
from .guards import hit_bounds_labels
from .integrators import integrate_radial_mass_grid
from .models import G_ASTRO, baryonic_velocity_sq, iso_density, iso_mass_exact, iso_velocity
from .orbits import euler_orbit, iso_potential, orbit_diagnostics, rk4_orbit, velocity_verlet_orbit
from .solvers import LOG_HI, LOG_LO, HaloFit, baryonic_only_fit, find_crossover_radius, fit_galaxy_halo, halo_model_velocity

ISO_PARAM_LABELS = ("rho0", "rc", "upsilon_d")
NFW_PARAM_LABELS = ("rhos", "rs", "upsilon_d")


def _dm_frac(galaxy: GalaxyData, fit: HaloFit) -> float:
    v_tot, v_dm = halo_model_velocity(galaxy, fit.model_kind, fit.rho0, fit.r_scale, fit.upsilon_disk)
    return float(np.clip(float(v_dm[-1] ** 2) / max(float(v_tot[-1] ** 2), 1e-12), 0.0, 1.0))


def baryon_mass_and_potential(
    galaxy: GalaxyData, upsilon_disk: float
) -> tuple[Callable[[float], float], Callable[[float], float]]:
    """Construct spherical-equivalent baryonic ``M(r) = r*max(v_bar^2, 0)/G`` and exact matching central potential."""
    r_tab = np.concatenate(([0.0], galaxy.rad_kpc))
    m_tab = np.concatenate(([0.0], galaxy.rad_kpc * np.maximum(baryonic_velocity_sq(galaxy, upsilon_disk), 0.0) / G_ASTRO))
    r_min, r_max = float(galaxy.rad_kpc[0]), float(r_tab[-1])
    m_min, m_max = float(m_tab[1]), float(m_tab[-1])
    mass_fn = lambda r: m_max if r >= r_max else float(np.interp(r, r_tab, m_tab))
    slopes = np.diff(m_tab[1:]) / np.diff(r_tab[1:])
    intercepts = m_tab[1:-1] - slopes * r_tab[1:-1]
    seg_dphi = G_ASTRO * (
        intercepts * (1.0 / r_tab[1:-1] - 1.0 / r_tab[2:]) + slopes * np.log(r_tab[2:] / r_tab[1:-1])
    )
    phi_nodes = -G_ASTRO * m_max / r_max - np.concatenate((np.cumsum(seg_dphi[::-1])[::-1], [0.0]))
    phi_min = float(phi_nodes[0])
    inner_slope = G_ASTRO * (m_min / r_min)

    def pot_fn(r: float) -> float:
        rf = float(r)
        if not np.isfinite(rf) or rf <= 0.0:
            raise ValueError("Radius for baryonic potential must be finite and strictly positive")
        if rf >= r_max:
            return -G_ASTRO * m_max / rf
        if rf <= r_min:
            return phi_min - inner_slope * float(np.log(r_min / rf))
        idx = int(np.searchsorted(r_tab, rf, side="right")) - 2
        ra = float(r_tab[idx + 1])
        return float(
            phi_nodes[idx]
            + G_ASTRO * (intercepts[idx] * (1.0 / ra - 1.0 / rf) + slopes[idx] * np.log(rf / ra))
        )

    return mass_fn, pot_fn


def _deep_dive(gal: GalaxyData, fb: HaloFit, fi: HaloFit, fn: HaloFit) -> dict[str, object]:
    try:
        bis, newt = find_crossover_radius(gal, fi.rho0, fi.r_scale, fi.upsilon_disk, "iso")
        crossover = {
            "dm_dominated_everywhere": False,
            "bisection_kpc": bis.root,
            "bisection_iters": bis.iterations,
            "newton_kpc": newt.root,
            "newton_iters": newt.iterations,
            "abs_diff_kpc": abs(bis.root - newt.root),
        }
    except ValueError:
        vals = iso_velocity(gal.rad_kpc, fi.rho0, fi.r_scale) ** 2 - baryonic_velocity_sq(gal, fi.upsilon_disk)
        crossover = {
            "dm_dominated_everywhere": bool(np.all(vals >= 0.0)),
            "bisection_kpc": None,
            "bisection_iters": None,
            "newton_kpc": None,
            "newton_iters": None,
            "abs_diff_kpc": None,
        }
    rho_fn = lambda r: iso_density(r, fi.rho0, fi.r_scale)
    m_exact = float(iso_mass_exact(gal.rad_kpc[-1], fi.rho0, fi.r_scale))
    m_simp = float(integrate_radial_mass_grid(rho_fn, gal.rad_kpc, "simpson", 32)[-1])
    m_gauss = float(integrate_radial_mass_grid(rho_fn, gal.rad_kpc, "gauss", 16)[-1])
    return {
        "baryons": {"upsilon_disk": fb.upsilon_disk, "chi2_red": fb.chi2_red},
        "iso": {
            "rho0": fi.rho0,
            "rc": fi.r_scale,
            "upsilon_disk": fi.upsilon_disk,
            "chi2_red": fi.chi2_red,
            "f_dm_outer": _dm_frac(gal, fi),
        },
        "nfw": {
            "rhos": fn.rho0,
            "rs": fn.r_scale,
            "upsilon_disk": fn.upsilon_disk,
            "chi2_red": fn.chi2_red,
            "f_dm_outer": _dm_frac(gal, fn),
        },
        "crossover": crossover,
        "enclosed_mass_outer_msun": {
            "exact": m_exact,
            "simpson": m_simp,
            "gauss": m_gauss,
            "simpson_rel_err": abs(m_simp - m_exact) / m_exact,
            "gauss_rel_err": abs(m_gauss - m_exact) / m_exact,
        },
    }


def run_survey(data_dir: str | Path | None = None, output_path: str | Path | None = None) -> dict[str, object]:
    """Run SPARC survey, deep dives (NGC3198/DDO154 if present), and NGC3198 outer-star orbit test."""
    repo_root = Path(__file__).resolve().parents[2]
    root = repo_root if (repo_root / "data").is_dir() else Path.cwd()
    target_dir = Path(data_dir) if data_dir is not None else (root / "data")
    catalog = load_sparc_catalog(target_dir, min_points=5)
    if not catalog:
        raise ValueError(f"No galaxies with >= 5 radial measurements found in {target_dir}")
    out_file = Path(output_path) if output_path is not None else (root / "outputs" / "sparc_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    rows, fits = [], {}
    for name, gal in catalog.items():
        fb, fi, fn = baryonic_only_fit(gal), fit_galaxy_halo(gal, "iso"), fit_galaxy_halo(gal, "nfw")
        fits[name] = (fb, fi, fn)
        iso_hits = hit_bounds_labels(fi.opt.x, (LOG_LO, LOG_HI), ISO_PARAM_LABELS)
        nfw_hits = hit_bounds_labels(fn.opt.x, (LOG_LO, LOG_HI), NFW_PARAM_LABELS)
        rows.append({
            "name": name,
            "n_points": int(gal.rad_kpc.size),
            "r_outer_kpc": float(gal.rad_kpc[-1]),
            "chi2_baryons": fb.chi2_red,
            "chi2_iso": fi.chi2_red,
            "chi2_nfw": fn.chi2_red,
            "baryons_converged": bool(fb.opt.converged),
            "baryons_iterations": int(fb.opt.iterations),
            "baryons_stop_reason": fb.opt.stop_reason,
            "iso_converged": bool(fi.opt.converged),
            "iso_iterations": int(fi.opt.iterations),
            "iso_grad_norm": fi.opt.grad_norm,
            "iso_stop_reason": fi.opt.stop_reason,
            "nfw_converged": bool(fn.opt.converged),
            "nfw_iterations": int(fn.opt.iterations),
            "nfw_grad_norm": fn.opt.grad_norm,
            "nfw_stop_reason": fn.opt.stop_reason,
            "f_dm_outer_iso": _dm_frac(gal, fi),
            "f_dm_outer_nfw": _dm_frac(gal, fn),
            "iso_rho0": fi.rho0,
            "iso_rc": fi.r_scale,
            "iso_upsilon_d": fi.upsilon_disk,
            "iso_bounds_hit": iso_hits,
            "nfw_rhos": fn.rho0,
            "nfw_rs": fn.r_scale,
            "nfw_upsilon_d": fn.upsilon_disk,
            "nfw_bounds_hit": nfw_hits,
        })

    orbit_test: dict[str, object] | None = None
    if "NGC3198" in catalog:
        ngc, (_, ngc_iso, _) = catalog["NGC3198"], fits["NGC3198"]
        r0_kpc, dt_gyr, n_steps = 25.0, 0.01, 200
        v0_kms = float(np.interp(r0_kpc, ngc.rad_kpc, ngc.v_obs))
        m_bar, p_bar = baryon_mass_and_potential(ngc, ngc_iso.upsilon_disk)
        m_tot = lambda r: m_bar(r) + float(iso_mass_exact(r, ngc_iso.rho0, ngc_iso.r_scale))
        p_tot = lambda r: p_bar(r) + iso_potential(r, ngc_iso.rho0, ngc_iso.r_scale)
        p_bar_r0 = p_bar(r0_kpc)
        orbit_test = {
            "galaxy": "NGC3198",
            "approximation": "central_force_spherical_equivalent_baryons",
            "upsilon_disk": ngc_iso.upsilon_disk,
            "baryons_only_v_esc_kms": float(np.sqrt(max(-2.0 * p_bar_r0, 0.0))),
            "baryons_only_specific_energy": float(0.5 * v0_kms**2 + p_bar_r0),
            "r0_kpc": r0_kpc,
            "v0_kms": v0_kms,
            "dt_gyr": dt_gyr,
            "n_steps": n_steps,
            "baryons_only_rk4": orbit_diagnostics(rk4_orbit(r0_kpc, v0_kms, m_bar, dt_gyr, n_steps), p_bar),
            "dark_halo_rk4": orbit_diagnostics(rk4_orbit(r0_kpc, v0_kms, m_tot, dt_gyr, n_steps), p_tot),
            "dark_halo_verlet": orbit_diagnostics(velocity_verlet_orbit(r0_kpc, v0_kms, m_tot, dt_gyr, n_steps), p_tot),
            "dark_halo_euler": orbit_diagnostics(euler_orbit(r0_kpc, v0_kms, m_tot, dt_gyr, n_steps), p_tot),
        }

    def _count_hits(key: str, labels: tuple[str, ...]) -> dict[str, int]:
        return {p: sum(1 for r in rows if any(h.startswith(f"{p}_") for h in r[key])) for p in labels}

    iso_wins = sum(1 for r in rows if r["chi2_iso"] < r["chi2_nfw"])
    nfw_wins = sum(1 for r in rows if r["chi2_nfw"] < r["chi2_iso"])
    iso_gnorms = [r["iso_grad_norm"] for r in rows if r["iso_grad_norm"] is not None]
    nfw_gnorms = [r["nfw_grad_norm"] for r in rows if r["nfw_grad_norm"] is not None]
    results: dict[str, object] = {
        "survey": {
            "n_galaxies": len(rows),
            "iso_wins": iso_wins,
            "nfw_wins": nfw_wins,
            "non_converged_baryons": sum(1 for r in rows if not r["baryons_converged"]),
            "non_converged_iso": sum(1 for r in rows if not r["iso_converged"]),
            "non_converged_nfw": sum(1 for r in rows if not r["nfw_converged"]),
            "iso_machine_floor_stops": sum(1 for r in rows if r["iso_stop_reason"] == "machine_floor"),
            "nfw_machine_floor_stops": sum(1 for r in rows if r["nfw_stop_reason"] == "machine_floor"),
            "median_grad_norm_iso": float(np.median(iso_gnorms)) if iso_gnorms else None,
            "max_grad_norm_iso": float(np.max(iso_gnorms)) if iso_gnorms else None,
            "median_grad_norm_nfw": float(np.median(nfw_gnorms)) if nfw_gnorms else None,
            "max_grad_norm_nfw": float(np.max(nfw_gnorms)) if nfw_gnorms else None,
            "iso_bounds_hits": _count_hits("iso_bounds_hit", ISO_PARAM_LABELS),
            "nfw_bounds_hits": _count_hits("nfw_bounds_hit", NFW_PARAM_LABELS),
            "median_chi2_baryons": float(np.median([r["chi2_baryons"] for r in rows])),
            "median_chi2_iso": float(np.median([r["chi2_iso"] for r in rows])),
            "median_chi2_nfw": float(np.median([r["chi2_nfw"] for r in rows])),
            "median_f_dm_outer_iso": float(np.median([r["f_dm_outer_iso"] for r in rows])),
            "median_f_dm_outer_nfw": float(np.median([r["f_dm_outer_nfw"] for r in rows])),
        },
        "deep_dives": {g: _deep_dive(catalog[g], *fits[g]) for g in ("NGC3198", "DDO154") if g in catalog},
        "orbit_test": orbit_test,
        "galaxies": rows,
    }
    out_file.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    return results


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the SPARC dark-matter halo survey and NGC 3198 orbit test.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing SPARC *_rotmod.dat files.")
    parser.add_argument("--output", type=Path, default=None, help="Output path for the JSON survey report.")
    args = parser.parse_args(argv)
    res = run_survey(data_dir=args.data_dir, output_path=args.output)
    s, o = res["survey"], res["orbit_test"]
    orbit_str = (
        f" | NGC3198 25kpc r_end/r0: Bar={o['baryons_only_rk4']['r_final_ratio']:.2f} "
        f"DM={o['dark_halo_rk4']['r_final_ratio']:.2f}"
        if o is not None
        else ""
    )
    print(
        f"[SPARC n={s['n_galaxies']}] med chi2_red: Bar={s['median_chi2_baryons']:.2f} "
        f"ISO={s['median_chi2_iso']:.2f} NFW={s['median_chi2_nfw']:.2f} | "
        f"ISO/NFW wins={s['iso_wins']}/{s['nfw_wins']} | f_DM={s['median_f_dm_outer_iso']:.2f}{orbit_str}"
    )


if __name__ == "__main__":
    main()
