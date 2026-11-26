"""Root finding and nonlinear optimization for SPARC halo fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .data import GalaxyData
from .guards import (
    BoundsPair,
    active_set_changed,
    check_cg_inputs,
    check_interval,
    check_newton_bounds,
    check_tol_iter,
    project_box_vector,
)
from .models import G_ASTRO, baryonic_velocity_sq, chi2_reduced, iso_velocity, nfw_velocity, total_velocity

GOLDEN = 0.5 * (3.0 - np.sqrt(5.0))  # ~0.381966
_EPS = float(np.finfo(np.float64).eps)
HALO_VELOCITY = {"iso": iso_velocity, "nfw": nfw_velocity}
LOG_LO = np.array([3.0, -1.0, -1.0])
LOG_HI = np.array([11.0, 3.0, np.log10(2.0)])


@dataclass(frozen=True)
class RootResult:
    root: float
    f_root: float
    iterations: int
    converged: bool
    trace: tuple[float, ...]


@dataclass(frozen=True)
class OptResult:
    x: float | np.ndarray
    fun: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class HaloFit:
    """Fitted halo/baryon parameters (``rho0``/``r_scale`` hold ``rho0, rc`` for ISO, ``rhos, rs`` for NFW)."""

    model_kind: str
    rho0: float
    r_scale: float
    upsilon_disk: float
    chi2_red: float
    opt: OptResult


def bisection(f: Callable[[float], float], a: float, b: float, tol: float = 1e-8, max_iter: int = 100) -> RootResult:
    """Bracketed bisection root finder on ``[a, b]`` with opposite-sign endpoints."""
    lo, hi, n_max = check_interval(a, b, tol, max_iter)
    fa, fb = float(f(lo)), float(f(hi))
    if not (np.isfinite(fa) and np.isfinite(fb)):
        raise ValueError(f"Non-finite bracket values: f({lo})={fa}, f({hi})={fb}")
    if fa == 0.0 or fb == 0.0:
        root = float(lo if fa == 0.0 else hi)
        return RootResult(root, 0.0, 0, True, (root,))
    if (fa > 0.0) == (fb > 0.0):
        raise ValueError(f"Root not bracketed: f({lo})={fa:.3g}, f({hi})={fb:.3g}")

    flo, trace = fa, []
    for k in range(1, n_max + 1):
        mid = 0.5 * (lo + hi)
        fm = float(f(mid))
        if not np.isfinite(fm):
            raise ValueError(f"Non-finite function value f({mid})={fm}")
        trace.append(mid)
        if fm == 0.0 or 0.5 * (hi - lo) <= tol:
            return RootResult(mid, fm, k, True, tuple(trace))
        lo, hi, flo = (lo, mid, flo) if (flo > 0.0) != (fm > 0.0) else (mid, hi, fm)
    return RootResult(mid, fm, n_max, False, tuple(trace))


def _root_witness(
    f: Callable[[float], float], x: float, fx: float, dfx: float, tol: float, bounds: tuple[float, float] | None
) -> bool:
    """Return True when ``f`` crosses zero within ``±max(tol, 2 ULP)`` of ``x`` with slope matching ``dfx``."""
    span = max(tol, 2.0 * _EPS * max(abs(x), 1.0))
    x_lo, x_hi = x - span, x + span
    if bounds is not None:
        x_lo, x_hi = max(bounds[0], x_lo), min(bounds[1], x_hi)
    flo, fhi = float(f(x_lo)), float(f(x_hi))
    if not (np.isfinite(flo) and np.isfinite(fhi)):
        return False
    sign_change = (
        flo == 0.0
        or fhi == 0.0
        or (flo > 0.0) != (fhi > 0.0)
        or (flo > 0.0) != (fx > 0.0)
        or (fhi > 0.0) != (fx > 0.0)
    )
    return sign_change and (fhi - flo) * dfx > 0.0


def newton_raphson(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-10,
    max_iter: int = 50,
    bounds: tuple[float, float] | None = None,
) -> RootResult:
    """Newton-Raphson root finder with derivative guard and optional bracketing interval safeguard."""
    n_max = check_tol_iter(tol, max_iter)
    b_lo, b_hi, has_bounds = check_newton_bounds(x0, bounds)
    bracketed, f_lo, f_hi = False, 0.0, 0.0
    if has_bounds:
        f_lo, f_hi = float(f(b_lo)), float(f(b_hi))
        if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
            raise ValueError(f"Non-finite bracket values: f({b_lo})={f_lo}, f({b_hi})={f_hi}")
        if f_lo == 0.0 or f_hi == 0.0:
            root = float(b_lo if f_lo == 0.0 else b_hi)
            return RootResult(root, 0.0, 0, True, (root,))
        if (f_lo > 0.0) == (f_hi > 0.0):
            raise ValueError(f"Root not bracketed in bounds [{b_lo}, {b_hi}]: f(lo)={f_lo:.3g}, f(hi)={f_hi:.3g}")
        bracketed = True

    x, trace = float(x0), [float(x0)]
    fx = float(f(x))
    if not np.isfinite(fx):
        return RootResult(x, fx, 0, False, tuple(trace))
    if fx == 0.0:
        return RootResult(x, 0.0, 0, True, tuple(trace))
    f_scale = min(1.0, max(abs(fx), abs(f_lo), abs(f_hi)) if bracketed else abs(fx))

    for k in range(1, n_max + 1):
        if bracketed:
            if (f_lo > 0.0) != (fx > 0.0):
                b_hi = x
            else:
                b_lo, f_lo = x, fx
        dfx = float(df(x))
        used_bisection = False
        if not np.isfinite(dfx) or abs(dfx) < 1e-14:
            if not bracketed:
                return RootResult(x, fx, k - 1, False, tuple(trace))
            x_next, used_bisection = 0.5 * (b_lo + b_hi), True
        else:
            dx = fx / dfx
            x_raw = x - dx
            if not np.isfinite(x_raw):
                if not bracketed:
                    return RootResult(x, fx, k - 1, False, tuple(trace))
                x_next, used_bisection = 0.5 * (b_lo + b_hi), True
            elif abs(dx) <= tol and (not bracketed or b_lo <= x_raw <= b_hi):
                fx_raw = float(f(x_raw))
                if not np.isfinite(fx_raw):
                    return RootResult(x_raw, fx_raw, k, False, tuple(trace) + (x_raw,))
                cur_bounds = (b_lo, b_hi) if bracketed else None
                if fx_raw == 0.0 or abs(fx_raw) <= tol * f_scale or _root_witness(f, x_raw, fx_raw, dfx, tol, cur_bounds):
                    trace.append(x_raw)
                    return RootResult(x_raw, fx_raw, k, True, tuple(trace))
                if bracketed and (x_raw <= b_lo or x_raw >= b_hi):
                    x_next, used_bisection = 0.5 * (b_lo + b_hi), True
                else:
                    x, fx = x_raw, fx_raw
                    trace.append(x)
                    continue
            elif bracketed and (x_raw <= b_lo or x_raw >= b_hi):
                x_next, used_bisection = 0.5 * (b_lo + b_hi), True
            else:
                x_next = x_raw

        x = x_next
        trace.append(x)
        fx = float(f(x))
        if not np.isfinite(fx):
            return RootResult(x, fx, k, False, tuple(trace))
        if fx == 0.0 or (used_bisection and 0.5 * (b_hi - b_lo) <= tol):
            return RootResult(x, fx, k, True, tuple(trace))
    return RootResult(x, fx, n_max, False, tuple(trace))


def brent_minimize(f: Callable[[float], float], a: float, b: float, tol: float = 1e-6, max_iter: int = 80) -> OptResult:
    """1D minimization via parabolic interpolation with golden-section fallback."""
    lo, hi, n_max = check_interval(a, b, tol, max_iter)
    x = w = v = lo + GOLDEN * (hi - lo)
    fx = fw = fv = float(f(x))
    if not np.isfinite(fx):
        return OptResult(x, fx, 0, False)

    step = prev = 0.0
    for k in range(1, n_max + 1):
        mid, tol1 = 0.5 * (lo + hi), tol * abs(x) + 1e-12
        if abs(x - mid) <= 2.0 * tol1 - 0.5 * (hi - lo):
            return OptResult(x, fx, k - 1, True)
        use_golden = True
        if abs(prev) > tol1:
            p = (x - w) ** 2 * (fx - fv) - (x - v) ** 2 * (fx - fw)
            q = (x - w) * (fx - fv) - (x - v) * (fx - fw)
            if q != 0.0:
                u = x - 0.5 * p / q
                if lo + 2.0 * tol1 <= u <= hi - 2.0 * tol1 and abs(u - x) < 0.5 * abs(prev):
                    prev, step, use_golden = step, u - x, False
        if use_golden:
            prev = (hi - x) if x < mid else (lo - x)
            step = GOLDEN * prev
        u = x + (step if abs(step) >= tol1 else float(np.copysign(tol1, step)))
        fu = float(f(u))
        if not np.isfinite(fu):
            return OptResult(x, fx, k, False)
        if fu <= fx:
            lo, hi = (lo, x) if u < x else (x, hi)
            v, fv, w, fw, x, fx = w, fw, x, fx, u, fu
        else:
            lo, hi = (u, hi) if u < x else (lo, u)
            if fu <= fw or w == x:
                v, fv, w, fw = w, fw, u, fu
            elif fu <= fv or v in (x, w):
                v, fv = u, fu
    return OptResult(x, fx, n_max, False)


def _gradient(f: Callable[[np.ndarray], float], x: np.ndarray, bounds: BoundsPair) -> np.ndarray:
    grad = np.empty(x.size, dtype=np.float64)
    for i in range(x.size):
        h = max(1e-5, 1e-11 * abs(float(x[i])))
        lo_i = float(bounds[0][i]) if bounds is not None else -np.inf
        hi_i = float(bounds[1][i]) if bounds is not None else np.inf
        x_lo, x_hi = max(lo_i, x[i] - h), min(hi_i, x[i] + h)
        p_lo, p_hi = x.copy(), x.copy()
        p_lo[i], p_hi[i] = x_lo, x_hi
        grad[i] = (float(f(p_hi)) - float(f(p_lo))) / (x_hi - x_lo)
    return grad


def _bracket_ray(phi: Callable[[float], float], f0: float, step_init: float, max_step: float) -> tuple[float, float, bool]:
    """Bracket a strict descent minimum along ``t >= 0``; returns ``(lo, hi, decreased)``."""
    a, b = 0.0, min(step_init, max_step)
    fb = float(phi(b))
    while (not np.isfinite(fb) or fb >= f0) and b > 1e-13:
        b *= 0.25
        fb = float(phi(b))
    if not np.isfinite(fb) or fb >= f0:
        return 0.0, b, False
    for _ in range(18):
        c = min(b + 1.61803398875 * (b - a), max_step)
        fc = float(phi(c))
        if not np.isfinite(fc):
            lo_f, hi_bad = b, c
            for _ in range(12):
                mid = 0.5 * (lo_f + hi_bad)
                fm = float(phi(mid))
                if np.isfinite(fm):
                    lo_f = mid
                    if fm > fb:
                        return a, mid, True
                    fb = fm
                else:
                    hi_bad = mid
            return a, lo_f, True
        if fc > fb or c == max_step:
            return a, c, True
        a, b, fb = b, c, fc
    return a, b, True


def _at_machine_floor(
    loss_fn: Callable[[np.ndarray], float], x: np.ndarray, g: np.ndarray, fx: float, tol: float, bounds: BoundsPair
) -> bool:
    """Confirm strict local convexity and float64-floor Newton decrement on free coordinates."""
    if float(np.linalg.norm(g)) >= 1e-3:
        return False
    free = np.ones(x.size, dtype=bool)
    if bounds is not None:
        lo, hi = bounds
        g_raw = _gradient(loss_fn, x, bounds)
        free = ~(((x <= lo + 1e-12) & (g_raw > 0.0)) | ((x >= hi - 1e-12) & (g_raw < 0.0)))
    idx = np.flatnonzero(free)
    if idx.size == 0:
        return True
    if idx.size > 10:
        return False
    m = idx.size
    H = np.empty((m, m), dtype=np.float64)
    for a, i in enumerate(idx):
        h = max(1e-4, 1e-10 * abs(float(x[i])))
        lo_i = float(bounds[0][i]) if bounds is not None else -np.inf
        hi_i = float(bounds[1][i]) if bounds is not None else np.inf
        x_lo, x_hi = max(lo_i, x[i] - h), min(hi_i, x[i] + h)
        p_lo, p_hi = x.copy(), x.copy()
        p_lo[i], p_hi[i] = x_lo, x_hi
        H[:, a] = (_gradient(loss_fn, p_hi, bounds)[idx] - _gradient(loss_fn, p_lo, bounds)[idx]) / (x_hi - x_lo)
    H = 0.5 * (H + H.T)
    if not np.all(np.isfinite(H)):
        return False
    evals, evecs = np.linalg.eigh(H)
    if float(evals[0]) <= 0.0:
        return False
    gf = g[idx]
    step = evecs @ ((evecs.T @ gf) / evals)
    df_pred = 0.5 * float(gf @ step)
    if df_pred > 1e-10 * max(1.0, abs(fx)) or float(np.linalg.norm(step)) > max(100.0 * tol, 1e-4):
        return False
    pt = x.copy()
    pt[idx] -= step
    if bounds is not None:
        pt = np.clip(pt, bounds[0], bounds[1])
    f_try = float(loss_fn(pt))
    return np.isfinite(f_try) and f_try >= fx - 1e-10 * max(1.0, abs(fx))


def conjugate_gradient(
    loss_fn: Callable[[np.ndarray], float],
    x0,
    tol: float = 1e-6,
    max_iter: int = 500,
    max_step: float = 10.0,
    bounds: BoundsPair = None,
) -> OptResult:
    """Projected Polak-Ribiere+ CG with finite-diff gradient, monotone Brent line search, and combined stopping rule.

    Converges when projected ``||g|| < tol``, or when line search stalls at a strictly convex float64-floor
    minimum verified by the free-coordinate Hessian and Newton decrement.
    """
    x, n_max, b_pair = check_cg_inputs(x0, tol, max_iter, max_step, bounds)
    fx = float(loss_fn(x))
    g = project_box_vector(x, _gradient(loss_fn, x, b_pair), b_pair, flip=1.0)
    if not (np.isfinite(fx) and np.all(np.isfinite(g))):
        return OptResult(x, fx, 0, False)

    d, steepest, step_hint, stall_count = -g.copy(), True, 0.5, 0
    restart_period = max(x.size * 2, 6)
    for k in range(1, n_max + 1):
        if float(np.linalg.norm(g)) < tol:
            return OptResult(x, fx, k - 1, True)
        d = project_box_vector(x, d, b_pair, flip=-1.0)
        if float(g @ d) >= 0.0 or float(np.linalg.norm(d)) < 1e-30:
            d, steepest = -g.copy(), True
        d_norm = float(np.linalg.norm(d))
        d_hat = d / d_norm
        phi = (
            (lambda t: float(loss_fn(np.clip(x + t * d_hat, b_pair[0], b_pair[1]))))
            if b_pair is not None
            else (lambda t: float(loss_fn(x + t * d_hat)))
        )
        lo, hi, decreased = _bracket_ray(phi, fx, step_hint, max_step)
        if not decreased:
            if not steepest:
                d, steepest, step_hint = -g.copy(), True, min(step_hint, 1e-2)
                continue
            converged = _at_machine_floor(loss_fn, x, g, fx, tol, b_pair)
            return OptResult(x, fx, k - 1 if converged else k, converged)
        line = brent_minimize(phi, lo, hi, tol=1e-9, max_iter=80)
        if not np.isfinite(line.fun) or float(line.fun) >= fx:
            if not steepest:
                d, steepest, step_hint = -g.copy(), True, min(step_hint, 1e-2)
                continue
            converged = _at_machine_floor(loss_fn, x, g, fx, tol, b_pair)
            return OptResult(x, fx, k - 1 if converged else k, converged)
        step_taken = float(line.x)
        step_hint = min(max_step, max(step_taken * 2.0, 1e-8))
        x_new = x + step_taken * d_hat
        if b_pair is not None:
            x_new = np.clip(x_new, b_pair[0], b_pair[1])
        f_new = float(line.fun)
        g_new = project_box_vector(x_new, _gradient(loss_fn, x_new, b_pair), b_pair, flip=1.0)
        if not np.all(np.isfinite(g_new)):
            return OptResult(x_new, f_new, k, False)
        gg = float(g @ g)
        if active_set_changed(x, x_new, b_pair) or gg <= 1e-30 or (k % restart_period == 0):
            beta = 0.0
        else:
            beta = max(0.0, float(g_new @ (g_new - g)) / gg)
        d = -g_new + beta * d
        dx_norm, df_abs = float(np.linalg.norm(x_new - x)), abs(fx - f_new)
        x, fx, g, steepest = x_new, f_new, g_new, (beta == 0.0)
        if float(np.linalg.norm(g)) < tol:
            return OptResult(x, fx, k, True)
        if dx_norm < 1e-9 and df_abs <= 1e-12 * max(1.0, abs(fx)):
            stall_count += 1
            if stall_count >= 2 and _at_machine_floor(loss_fn, x, g, fx, tol, b_pair):
                return OptResult(x, fx, k, True)
        else:
            stall_count = 0
    return OptResult(x, fx, n_max, False)


def halo_model_velocity(
    galaxy: GalaxyData, model_kind: str, rho0: float, r_scale: float, upsilon_disk: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(v_total, v_dm)`` in km/s at the galaxy's observed radii."""
    if model_kind not in HALO_VELOCITY:
        raise ValueError(f"Unsupported model_kind: {model_kind!r}")
    v_dm = HALO_VELOCITY[model_kind](galaxy.rad_kpc, rho0, r_scale)
    return total_velocity(baryonic_velocity_sq(galaxy, upsilon_disk), v_dm), v_dm


def fit_galaxy_halo(galaxy: GalaxyData, model_kind: str = "iso", x0: np.ndarray | None = None) -> HaloFit:
    """Fit ``[log10 rho, log10 r_scale, log10 upsilon_disk]`` via multi-start projected CG (4 ISO / 3 NFW starts)."""
    if model_kind not in HALO_VELOCITY:
        raise ValueError(f"Unsupported model_kind: {model_kind!r}")
    inv_err = 1.0 / galaxy.v_err
    inv_dof = 1.0 / max(galaxy.rad_kpc.size - 3, 1)

    def loss(p: np.ndarray) -> float:
        vm, _ = halo_model_velocity(galaxy, model_kind, *(10.0**p))
        return float(np.sum(((galaxy.v_obs - vm) * inv_err) ** 2) * inv_dof)

    shape = 1.0 - np.arctan(1.0) if model_kind == "iso" else np.log(2.0) - 0.5
    v_ref, r_out = max(float(np.max(galaxy.v_obs)), 10.0), float(galaxy.rad_kpc[-1])
    if x0 is not None:
        starts = [np.asarray(x0, dtype=np.float64)]
    else:
        r_large = max(5.0 * r_out, 30.0)
        starts = [
            np.clip(np.log10([v_ref**2 / (4.0 * np.pi * G_ASTRO * r_s**2 * shape), r_s, 0.5]), LOG_LO, LOG_HI)
            for r_s in (max(r_out * f, 0.3) for f in (0.25, 0.7))
        ]
        if model_kind == "iso":
            r_small = max(min(r_out * 0.04, 0.5), 0.15)
            starts.append(np.clip(np.log10([v_ref**2 / (4.0 * np.pi * G_ASTRO * r_small**2 * shape), r_small, 0.25]), LOG_LO, LOG_HI))
            rho_large = 3.0 * v_ref**2 / (4.0 * np.pi * G_ASTRO * r_out**2)
            starts.append(np.clip(np.log10([rho_large, r_large, 0.5]), LOG_LO, LOG_HI))
        else:
            rho_large = v_ref**2 / (2.0 * np.pi * G_ASTRO * r_large * r_out)
            starts.append(np.clip(np.log10([rho_large, r_large, 0.5]), LOG_LO, LOG_HI))
    best = min(
        (conjugate_gradient(loss, s, max_iter=500, max_step=4.0, bounds=(LOG_LO, LOG_HI)) for s in starts),
        key=lambda r: (not r.converged, r.fun),
    )
    rho0, r_scale, ups = (float(v) for v in 10.0**best.x)
    vm, _ = halo_model_velocity(galaxy, model_kind, rho0, r_scale, ups)
    return HaloFit(model_kind, rho0, r_scale, ups, chi2_reduced(galaxy.v_obs, galaxy.v_err, vm, 3), best)


def baryonic_only_fit(galaxy: GalaxyData) -> HaloFit:
    """Fit zero-dark-matter model minimizing disk M/L ``upsilon_disk`` in [0.1, 2.0] via Brent."""
    inv_err = 1.0 / galaxy.v_err
    inv_dof = 1.0 / max(galaxy.rad_kpc.size - 1, 1)
    opt = brent_minimize(
        lambda u: float(np.sum(((galaxy.v_obs - total_velocity(baryonic_velocity_sq(galaxy, u), 0.0)) * inv_err) ** 2) * inv_dof),
        0.1,
        2.0,
    )
    ups = float(opt.x)
    vm = total_velocity(baryonic_velocity_sq(galaxy, ups), 0.0)
    return HaloFit("baryons", 0.0, float("nan"), ups, chi2_reduced(galaxy.v_obs, galaxy.v_err, vm, 1), opt)


def find_crossover_radius(
    galaxy: GalaxyData, rho0: float, rc: float, upsilon_disk: float, model_kind: str = "iso", tol: float = 1e-10
) -> tuple[RootResult, RootResult]:
    """Solve ``v_dm^2(r) = v_bar^2(r)`` via bisection and safeguarded Newton at shared ``tol`` within ``galaxy.rad_kpc``."""
    if model_kind not in HALO_VELOCITY:
        raise ValueError(f"Unsupported model_kind: {model_kind!r}")
    r_tab, vb2_tab = galaxy.rad_kpc, baryonic_velocity_sq(galaxy, upsilon_disk)
    r_min, r_max, v_halo = float(r_tab[0]), float(r_tab[-1]), HALO_VELOCITY[model_kind]
    vals = v_halo(r_tab, rho0, rc) ** 2 - vb2_tab
    up_idx: int | None = None
    last_sign = 0
    for i, val in enumerate(vals):
        sign = 1 if val > 0.0 else (-1 if val < 0.0 else 0)
        if sign > 0 and (last_sign < 0 or (last_sign == 0 and i > 0)):
            up_idx = i - 1
        elif sign == 0 and i == vals.size - 1 and last_sign < 0:
            up_idx = i - 1
        if sign != 0:
            last_sign = sign
    if up_idx is None:
        raise ValueError(f"{galaxy.name}: no baryon->dark-matter crossover in observed range [{r_min}, {r_max}] kpc")
    lo, hi = float(r_tab[up_idx]), float(r_tab[up_idx + 1])
    f = lambda r: float(v_halo(r, rho0, rc) ** 2 - np.interp(r, r_tab, vb2_tab))

    def df(r: float) -> float:
        h = min(max(1e-5 * r, 1e-6), 0.25 * (hi - lo))
        r_lo, r_hi = max(lo, r - h), min(hi, r + h)
        return (f(r_hi) - f(r_lo)) / (r_hi - r_lo)

    return bisection(f, lo, hi, tol=tol), newton_raphson(f, df, 0.5 * (lo + hi), tol=tol, bounds=(lo, hi))
