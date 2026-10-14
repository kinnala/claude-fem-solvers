"""Verification of the plane-stress elastoplastic solver.

(a) Rectangular strip 10 x 2 (quarter model with the same BC names as the
    specimen). Plane stress with free lateral edges gives a homogeneous
    uniaxial stress state, so the response must follow the exact bilinear law
        sigma = E eps                         (eps <= eps_y = sigma_y0 / E),
        sigma = sigma_y0 + E_t (eps - eps_y), E_t = E H / (E + H),
    elastic unloading with slope E, and the permanent strain
    eps_max - sigma_max / E. Newton must converge quadratically.
(b) Coarse dog-bone run: initial slope F/dL = E A_gauge / 50, F_max equal to
    the gauge-section analytic value within 2 %, permanent dL > 0.
"""
import logging

import numpy as np
import pytest
from skfem import MeshTri

import solver

logging.getLogger("skfem").setLevel(logging.ERROR)

P = solver.PARAMS
E, H, SY0, T = P["E"], P["H"], P["sigma_y0"], P["t"]
ET = E * H / (E + H)


def sigma_uniaxial(eps):
    """Monotonic uniaxial stress for linear isotropic hardening."""
    eps_y = SY0 / E
    return np.where(eps <= eps_y, E * eps, SY0 + ET * (eps - eps_y))


@pytest.fixture(scope="module")
def strip_run():
    mesh = MeshTri.init_tensor(np.linspace(0, 10, 11),
                               np.linspace(0, 2, 3)).with_boundaries({
        "sym_x": lambda x: np.isclose(x[0], 0.0),
        "sym_y": lambda x: np.isclose(x[1], 0.0),
        "grip": lambda x: np.isclose(x[0], 10.0),
        "free": lambda x: np.isclose(x[1], 2.0),
    })
    p = dict(P, u_max=0.1, n_load=10, n_unload=4)  # eps_max = 1 %
    return solver.assemble_and_solve(mesh, p, ext_x=5.0, verbose=False)


def test_strip_matches_bilinear_law(strip_run):
    model, hist, max_snap, final_snap = strip_run
    n = model.p["n_load"]
    area = 2.0 * T  # half width 2 times thickness; F = 2 t sum(reactions)
    eps = np.array([h["dL"] for h in hist]) / 10.0  # dL over 2 * ext_x
    F = np.array([h["F"] for h in hist])

    # loading branch
    F_ex = 2 * area * sigma_uniaxial(eps[:n + 1])
    assert np.allclose(F[:n + 1], F_ex, rtol=1e-6, atol=1e-6 * F_ex.max())

    # stress field at max load: homogeneous uniaxial
    s_max = sigma_uniaxial(eps[n])
    sig = max_snap["sig"]
    assert np.allclose(sig[0], s_max, rtol=1e-6)
    assert np.allclose(sig[1:], 0.0, atol=1e-6 * s_max)

    # elastic unloading branch and permanent strain
    F_un = 2 * area * (s_max - E * (eps[n] - eps[n:]))
    assert np.allclose(F[n:], F_un, rtol=1e-6, atol=1e-6 * F_un.max())
    eps_perm = eps[n] - s_max / E
    assert abs(final_snap["F"]) < 1e-6 * F.max()
    assert final_snap["dL"] / 10.0 == pytest.approx(eps_perm, rel=1e-6)


def test_newton_quadratic(strip_run):
    _, hist, _, _ = strip_run
    rates = []
    for h in hist:
        r = np.array(h["res"])
        if len(r) < 3:
            continue
        r = r[r > 1e-9 * r[0]]  # drop the round-off level
        for k in range(1, len(r) - 1):
            rates.append(np.log(r[k + 1] / r[k]) / np.log(r[k] / r[k - 1]))
    assert rates, "no plastic step with several Newton iterations"
    assert min(rates) > 1.7, rates


def test_dogbone_coarse(tmp_path):
    msh = solver.build_mesh(str(tmp_path / "coarse.msh"), h=3.0, h_fillet=1.5)
    mesh = solver.load_mesh(msh)
    p = dict(P, n_load=20, n_unload=4)  # first step elastic
    model, hist, max_snap, final_snap = solver.assemble_and_solve(
        mesh, p, verbose=False)
    a_gauge = P["W"] * T

    # initial (elastic) slope
    k_el = E * a_gauge / P["gauge"]
    assert hist[1]["F"] / hist[1]["dL"] == pytest.approx(k_el, rel=0.02)

    # max force against the gauge-section analytic value
    F_max = max(h["F"] for h in hist)
    eps_g = max_snap["dL"] / P["gauge"]
    F_an = sigma_uniaxial(eps_g) * a_gauge
    assert F_max == pytest.approx(F_an, rel=0.02)

    # unloaded to zero force with a permanent elongation
    assert abs(final_snap["F"]) < 1e-6 * F_max
    assert final_snap["dL"] > 0.0
