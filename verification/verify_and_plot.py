"""Convergence checks and 4-panel SPARC summary figure generator."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sparc_dm.cli import baryon_mass_and_potential, run_survey
from sparc_dm.data import load_galaxy
from sparc_dm.integrators import simpson_integrate, trapezoid_integrate
from sparc_dm.models import G_ASTRO, baryonic_velocity_sq, iso_mass_exact, iso_velocity, total_velocity
from sparc_dm.orbits import KPC_PER_GYR_PER_KMS, euler_orbit, rk4_orbit, velocity_verlet_orbit
from sparc_dm.solvers import halo_model_velocity


def _quadrature_and_ode_convergence() -> dict[str, object]:
    """Measure log-log convergence slopes for quadrature and Kepler orbit solvers."""
    rho0, rc, r_max = 4.5e7, 2.3, 6.0
    x_max = r_max / rc
    m_exact = 4.0 * math.pi * rho0 * (rc**3) * (x_max - math.atan(x_max))

    n_panels = np.array([20, 40, 80, 160], dtype=int)
    h_norm_q = n_panels[0] / n_panels.astype(np.float64)
    err_trap, err_simp = [], []
    for n in n_panels:
        r = np.linspace(0.0, r_max, int(n) + 1, dtype=np.float64)
        h = float(r[1] - r[0])
        integrand = 4.0 * math.pi * (r**2) * (rho0 / (1.0 + (r / rc) ** 2))
        err_trap.append(abs(trapezoid_integrate(integrand, h) - m_exact) / m_exact)
        err_simp.append(abs(simpson_integrate(integrand, h) - m_exact) / m_exact)

    m_pt, r0 = 1.0e11, 10.0
    v_circ = math.sqrt(G_ASTRO * m_pt / r0)
    t_orb = 2.0 * math.pi * r0 / (v_circ * KPC_PER_GYR_PER_KMS)
    n_steps = np.array([160, 320, 640, 1280], dtype=int)
    n_steps_euler = np.array([640, 1280, 2560, 5120], dtype=int)
    h_norm_o = n_steps[0] / n_steps.astype(np.float64)
    err_euler, err_verlet, err_rk4 = [], [], []
    for n, n_e in zip(n_steps, n_steps_euler):
        dt = t_orb / int(n)
        tr_e = euler_orbit(r0, v_circ, lambda _r: m_pt, t_orb / int(n_e), int(n_e))
        tr_v = velocity_verlet_orbit(r0, v_circ, lambda _r: m_pt, dt, int(n))
        tr_r = rk4_orbit(r0, v_circ, lambda _r: m_pt, dt, int(n))
        err_euler.append(float(np.hypot(tr_e[-1, 0] - r0, tr_e[-1, 1])) / r0)
        err_verlet.append(float(np.hypot(tr_v[-1, 0] - r0, tr_v[-1, 1])) / r0)
        err_rk4.append(float(np.hypot(tr_r[-1, 0] - r0, tr_r[-1, 1])) / r0)

    return {
        "h_norm_q": h_norm_q,
        "err_trap": np.array(err_trap),
        "err_simp": np.array(err_simp),
        "slope_trap": float(np.polyfit(np.log(h_norm_q), np.log(err_trap), 1)[0]),
        "slope_simp": float(np.polyfit(np.log(h_norm_q), np.log(err_simp), 1)[0]),
        "h_norm_o": h_norm_o,
        "err_euler": np.array(err_euler),
        "err_verlet": np.array(err_verlet),
        "err_rk4": np.array(err_rk4),
        "slope_euler": float(np.polyfit(np.log(h_norm_o), np.log(err_euler), 1)[0]),
        "slope_verlet": float(np.polyfit(np.log(h_norm_o), np.log(err_verlet), 1)[0]),
        "slope_rk4": float(np.polyfit(np.log(h_norm_o), np.log(err_rk4), 1)[0]),
    }


def verify_and_plot(output_dir: Path | str | None = None) -> Path:
    """Run SPARC survey, run convergence checks, and render 300-DPI 4-panel summary."""
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(output_dir) if output_dir is not None else root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    survey_data = run_survey(root / "data", out_dir / "sparc_results.json")

    ngc = load_galaxy(root / "data" / "NGC3198_rotmod.dat")
    dd = survey_data["deep_dives"]["NGC3198"]
    ngc_bar, ngc_iso, ngc_nfw, cross = dd["baryons"], dd["iso"], dd["nfw"], dd["crossover"]

    r0_kpc, dt_gyr, n_steps = 25.0, 0.01, 200
    v0_kms = float(np.interp(r0_kpc, ngc.rad_kpc, ngc.v_obs))
    m_bar, _ = baryon_mass_and_potential(ngc, ngc_iso["upsilon_disk"])
    m_tot = lambda r: m_bar(r) + float(iso_mass_exact(r, ngc_iso["rho0"], ngc_iso["rc"]))
    tr_bar = rk4_orbit(r0_kpc, v0_kms, m_bar, dt_gyr, n_steps)
    tr_dm_rk4 = rk4_orbit(r0_kpc, v0_kms, m_tot, dt_gyr, n_steps)
    tr_dm_euler = euler_orbit(r0_kpc, v0_kms, m_tot, dt_gyr, n_steps)

    conv = _quadrature_and_ode_convergence()

    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.4,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.2), dpi=300)

    # --- Panel A: NGC 3198 Rotation Curve (Baryons-only vs ISO vs NFW + Crossover) ---
    ax = axes[0, 0]
    r = ngc.rad_kpc
    v_bar_only = total_velocity(baryonic_velocity_sq(ngc, ngc_bar["upsilon_disk"]), 0.0)
    v_bar_iso = total_velocity(baryonic_velocity_sq(ngc, ngc_iso["upsilon_disk"]), 0.0)
    v_tot_iso, v_dm_iso = halo_model_velocity(ngc, "iso", ngc_iso["rho0"], ngc_iso["rc"], ngc_iso["upsilon_disk"])
    v_tot_nfw, _ = halo_model_velocity(ngc, "nfw", ngc_nfw["rhos"], ngc_nfw["rs"], ngc_nfw["upsilon_disk"])

    ax.errorbar(r, ngc.v_obs, yerr=ngc.v_err, fmt="o", color="#222222", ms=3.5, elinewidth=0.9, capsize=2, zorder=5, label="SPARC Observed $v_{\\mathrm{obs}}$")
    ax.plot(r, v_bar_only, "--", color="#c0392b", lw=1.7, label=f"Baryons-only ($\\chi^2_\\nu={ngc_bar['chi2_red']:.1f}$)")
    ax.plot(r, v_tot_iso, "-", color="#1f5a8a", lw=2.2, label=f"Baryons + ISO Core ($\\chi^2_\\nu={ngc_iso['chi2_red']:.2f}$)")
    ax.plot(r, v_tot_nfw, "-.", color="#d97706", lw=1.7, label=f"Baryons + NFW Cusp ($\\chi^2_\\nu={ngc_nfw['chi2_red']:.2f}$)")
    ax.plot(r, v_bar_iso, ":", color="#6b7280", lw=1.5, label=f"ISO fit: Baryons ($\\Upsilon_d={ngc_iso['upsilon_disk']:.2f}$)")
    ax.plot(r, v_dm_iso, linestyle=(0, (3, 1.5, 1, 1.5)), color="#0f766e", lw=1.6, label="ISO fit: Dark Halo")

    r_cross = float(cross["bisection_kpc"])
    v_cross = float(iso_velocity(r_cross, ngc_iso["rho0"], ngc_iso["rc"]))
    ax.axvline(r_cross, color="#0f766e", ls="--", lw=1.1, alpha=0.8)
    ax.scatter([r_cross], [v_cross], s=48, color="#0f766e", edgecolor="white", linewidth=0.9, zorder=6)
    ax.annotate(
        f"Crossover $r_\\times = {r_cross:.2f}$ kpc\n(Bisect {cross['bisection_iters']} vs Newton {cross['newton_iters']} it)",
        xy=(r_cross, v_cross),
        xytext=(r_cross + 4.0, v_cross + 62.0),
        fontsize=8.3,
        bbox={"boxstyle": "round,pad=0.3", "fc": "#f0fdfa", "ec": "#0f766e", "alpha": 0.92},
        arrowprops={"arrowstyle": "->", "color": "#0f766e", "lw": 1.1},
    )
    ax.set_xlim(0.0, 46.5)
    ax.set_ylim(0.0, 205.0)
    ax.set_xlabel("Galactocentric radius $r$ (kpc)")
    ax.set_ylabel("Circular velocity $v(r)$ (km/s)")
    ax.set_title("A. NGC 3198 Rotation Curve & Dark-Matter Crossover")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, ncol=1)
    ax.grid(alpha=0.25)

    # --- Panel B: 171-Galaxy SPARC Core vs Cusp chi^2 & Dark-Matter Fraction ---
    ax = axes[0, 1]
    gals = survey_data["galaxies"]
    chi_iso = np.array([g["chi2_iso"] for g in gals], dtype=np.float64)
    chi_nfw = np.array([g["chi2_nfw"] for g in gals], dtype=np.float64)
    f_dm = np.array([g["f_dm_outer_iso"] for g in gals], dtype=np.float64)

    sc = ax.scatter(chi_iso, chi_nfw, c=f_dm, cmap="viridis", vmin=0.0, vmax=1.0, s=26, alpha=0.85, edgecolors="#1f2937", linewidths=0.35, zorder=4)
    ax.plot([1.0e-2, 1.5e2], [1.0e-2, 1.5e2], "--", color="#4b5563", lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.0e-2, 1.5e2)
    ax.set_ylim(1.0e-2, 1.5e2)

    s = survey_data["survey"]
    n_non_conv = int(s["non_converged_baryons"]) + int(s["non_converged_iso"]) + int(s["non_converged_nfw"])
    conv_tag = "all converged" if n_non_conv == 0 else f"{n_non_conv} non-converged"
    box_text = (
        f"SPARC Survey ($N = {s['n_galaxies']}$, {conv_tag})\n"
        f"ISO Core wins: {s['iso_wins']}/{s['n_galaxies']} ({100.0 * s['iso_wins'] / s['n_galaxies']:.1f}%)\n"
        f"Med $\\chi^2_\\nu$: Bar {s['median_chi2_baryons']:.1f} | ISO {s['median_chi2_iso']:.2f} | NFW {s['median_chi2_nfw']:.2f}\n"
        f"Median outer $f_{{\\mathrm{{DM}}}} = {s['median_f_dm_outer_iso']:.2f}$"
    )
    ax.text(
        0.04, 0.96, box_text, transform=ax.transAxes, va="top", ha="left", fontsize=8.3, zorder=7,
        bbox={"boxstyle": "round,pad=0.35", "fc": "#ffffff", "ec": "#9ca3af", "alpha": 0.94},
    )
    ax.text(0.04, 0.71, "Above diagonal:\nISO Core fits better", transform=ax.transAxes, ha="left", va="top", fontsize=8.3, color="#1f5a8a", style="italic", zorder=7)
    ax.text(0.96, 0.05, "Below diagonal: NFW Cusp fits better", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.3, color="#b45309", style="italic", zorder=7)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("ISO outer dark-matter fraction $f_{\\mathrm{DM}}(R_{\\max})$", fontsize=9.0)
    ax.set_xlabel("ISO Core reduced $\\chi^2_\\nu$")
    ax.set_ylabel("NFW Cusp reduced $\\chi^2_\\nu$")
    ax.set_title("B. SPARC Survey: Core vs Cusp $\\chi^2_\\nu$ & Dark Fraction")
    ax.grid(True, which="both", alpha=0.22)

    # --- Panel C: Convergence Orders (Simpson O(h^4) vs Trap O(h^2) & RK4 vs Verlet vs Euler) ---
    ax = axes[1, 0]
    ax.loglog(conv["h_norm_q"], conv["err_trap"], "s--", color="#d97706", lw=1.7, ms=5.5, label=f"Trapezoid mass (slope = {conv['slope_trap']:.2f})")
    ax.loglog(conv["h_norm_q"], conv["err_simp"], "o-", color="#1f5a8a", lw=2.0, ms=5.5, label=f"Simpson mass (slope = {conv['slope_simp']:.2f})")
    ax.loglog(conv["h_norm_o"], conv["err_euler"], "d:", color="#c0392b", lw=1.7, ms=5.5, label=f"Euler orbit (slope = {conv['slope_euler']:.2f})")
    ax.loglog(conv["h_norm_o"], conv["err_verlet"], "^--", color="#0f766e", lw=1.7, ms=5.5, label=f"Verlet orbit (slope = {conv['slope_verlet']:.2f})")
    ax.loglog(conv["h_norm_o"], conv["err_rk4"], "v-", color="#7c3aed", lw=2.0, ms=5.5, label=f"RK4 orbit (slope = {conv['slope_rk4']:.2f})")

    ax.set_xlabel("Normalized step size $h / h_0$ (or $\\Delta t / \\Delta t_0$)")
    ax.set_ylabel("Relative error vs analytical solution")
    ax.set_title("C. Quadrature & Orbit ODE Convergence Slopes")
    ax.set_xlim(0.10, 1.15)
    ax.set_ylim(1e-13, 1e1)
    ax.set_xticks([0.1, 0.2, 0.5, 1.0])
    ax.set_xticklabels(["0.1", "0.2", "0.5", "1.0"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(loc="lower right", frameon=True, framealpha=0.94, fontsize=8.0)
    ax.grid(True, which="both", alpha=0.25)

    # --- Panel D: NGC 3198 2D Star Orbit at r0 = 25 kpc (Bound with DM vs Halo Removed) ---
    ax = axes[1, 1]
    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    ax.plot(r0_kpc * np.cos(theta), r0_kpc * np.sin(theta), ":", color="#9ca3af", lw=1.1, label=f"Initial circle $r_0 = {r0_kpc:.0f}$ kpc")
    ax.plot(tr_dm_rk4[:, 0], tr_dm_rk4[:, 1], "-", color="#1f5a8a", lw=2.2, label="Baryons + ISO Halo (RK4, bound)")
    ax.plot(tr_dm_euler[:, 0], tr_dm_euler[:, 1], "--", color="#d97706", lw=1.5, label="Baryons + ISO Halo (Euler drift)")
    ax.plot(tr_bar[:, 0], tr_bar[:, 1], "-.", color="#c0392b", lw=2.1, label=f"Halo removed ($\\Upsilon_d={ngc_iso['upsilon_disk']:.2f}$, RK4, escapes)")
    ax.scatter([0.0], [0.0], s=55, marker="*", color="#111827", zorder=6, label="NGC 3198 center")
    ax.scatter([r0_kpc], [0.0], s=42, color="#0f766e", edgecolor="white", linewidth=0.8, zorder=7)

    ax.annotate(
        f"Start $({r0_kpc:.0f}, 0)$ kpc\n$v_0 = {v0_kms:.1f}$ km/s",
        xy=(r0_kpc, 0.0),
        xytext=(34.0, 22.0),
        fontsize=8.3,
        bbox={"boxstyle": "round,pad=0.28", "fc": "#f0fdfa", "ec": "#0f766e", "alpha": 0.92},
        arrowprops={"arrowstyle": "->", "color": "#0f766e", "lw": 1.1},
    )
    r_bar = np.hypot(tr_bar[:, 0], tr_bar[:, 1])
    hits = np.flatnonzero(r_bar >= 75.0)
    esc_idx = int(hits[0]) if hits.size else r_bar.size - 1
    ax.annotate(
        f"Escapes ($r_{{2\\,\\mathrm{{Gyr}}}}/r_0 = {r_bar[-1] / r0_kpc:.1f}$)",
        xy=(tr_bar[esc_idx, 0], tr_bar[esc_idx, 1]),
        xytext=(-70.0, 58.0),
        fontsize=8.3,
        bbox={"boxstyle": "round,pad=0.28", "fc": "#fef2f2", "ec": "#c0392b", "alpha": 0.92},
        arrowprops={"arrowstyle": "->", "color": "#c0392b", "lw": 1.1},
    )

    ax.set_xlim(-78.0, 92.0)
    ax.set_ylim(-68.0, 88.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$x$ (kpc)")
    ax.set_ylabel("$y$ (kpc)")
    ax.set_title(f"D. NGC 3198 Central-Force Orbit ($2.0$ Gyr, $r_0 = {r0_kpc:.0f}$ kpc)")
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=8.0)
    ax.grid(alpha=0.25)

    fig.tight_layout(pad=1.6)
    fig_path = out_dir / "sparc_summary.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    return fig_path


def main() -> None:
    fig_path = verify_and_plot()
    print(f"Wrote summary figure -> {fig_path}")


if __name__ == "__main__":
    main()
