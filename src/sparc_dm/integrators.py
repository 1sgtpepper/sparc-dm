"""Numerical quadrature routines implemented with NumPy."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .guards import check_1d_grid, check_radial_inputs, require_int


def trapezoid_integrate(y: np.ndarray, h: float) -> float:
    """Composite trapezoidal rule on uniformly spaced samples with step ``h``."""
    arr = check_1d_grid(y, 2, h, "trapezoid_integrate")
    return float(h * (0.5 * (arr[0] + arr[-1]) + np.sum(arr[1:-1])))


def simpson_integrate(y: np.ndarray, h: float) -> float:
    """Composite Simpson's 1/3 rule on uniformly spaced samples with even interval count."""
    arr = check_1d_grid(y, 3, h, "simpson_integrate", require_even_intervals=True)
    return float((h / 3.0) * (arr[0] + arr[-1] + 4.0 * np.sum(arr[1:-1:2]) + 2.0 * np.sum(arr[2:-2:2])))


def gauss_legendre(n: int, a: float = -1.0, b: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``n``-point Gauss-Legendre nodes and weights mapped to ``[a, b]``."""
    n_int = require_int(n, 1, "n")
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError("Integration bounds a and b must be finite")

    m = (n_int + 1) // 2
    x = np.cos(np.pi * (np.arange(1, m + 1, dtype=np.float64) - 0.25) / (n_int + 0.5))

    for _ in range(30):
        p_prev, p_curr = np.ones_like(x), x.copy()
        for k in range(2, n_int + 1):
            p_prev, p_curr = p_curr, ((2.0 * k - 1.0) * x * p_curr - (k - 1.0) * p_prev) / k
        dp = n_int * (x * p_curr - p_prev) / (x * x - 1.0)
        dx = p_curr / dp
        x -= dx
        if np.max(np.abs(dx)) < 1e-15:
            break

    w = 2.0 / ((1.0 - x * x) * dp * dp)
    nodes, weights = np.empty(n_int, dtype=np.float64), np.empty(n_int, dtype=np.float64)
    nodes[:m], nodes[n_int - m :] = -x, x[::-1]
    weights[:m], weights[n_int - m :] = w, w[::-1]
    if n_int % 2 == 1:
        nodes[m - 1] = 0.0

    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    return mid + half * nodes, half * weights


def integrate_radial_mass_grid(
    rho_fn: Callable[[np.ndarray], np.ndarray],
    r_grid: np.ndarray,
    method: str = "gauss",
    n_nodes: int = 16,
) -> np.ndarray:
    """Cumulative enclosed mass ``M(R) = int_0^R 4*pi*r^2*rho(r) dr`` on ``r_grid``.

    For ``method="simpson"``, the ``r = 0`` endpoint is set to ``0.0`` (valid when ``r^2 * rho(r) -> 0`` as
    ``r -> 0``, as for ISO and NFW halos; use open ``method="gauss"`` for ``1/r^2`` singular profiles).
    """
    radii = check_radial_inputs(r_grid, method, n_nodes)
    if method == "gauss":
        ref_x, ref_w = gauss_legendre(n_nodes, 0.0, 1.0)

    order = np.argsort(radii)
    edges = np.concatenate(([0.0], radii[order]))
    panels = np.zeros(radii.size, dtype=np.float64)

    for i in range(radii.size):
        r0, r1 = edges[i], edges[i + 1]
        if r1 <= r0:
            continue
        if method == "gauss":
            pts = r0 + (r1 - r0) * ref_x
            panels[i] = (r1 - r0) * float(np.dot(ref_w, 4.0 * np.pi * pts * pts * np.asarray(rho_fn(pts), dtype=np.float64)))
        else:
            pts = np.linspace(r0, r1, int(n_nodes) + 1)
            integrand = np.zeros_like(pts)
            pos = pts > 0.0
            integrand[pos] = 4.0 * np.pi * (pts[pos] ** 2) * np.asarray(rho_fn(pts[pos]), dtype=np.float64)
            panels[i] = simpson_integrate(integrand, (r1 - r0) / int(n_nodes))

    out = np.empty_like(panels)
    out[order] = np.cumsum(panels)
    return out
