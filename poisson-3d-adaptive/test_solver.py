"""Verification of the adaptive solver.

(a) Manufactured solution u* = sin(pi x) sin(pi y) (1 + z) with the same
    k = x y z + 1, f = -div(k grad u*) and Neumann data g = k grad u* . n on
    front/back. Under uniform refinement the energy error must decay like
    N^{-2/3} (P2) and the effectivity index eta / ||k^{1/2} grad(u* - u_h)||
    must stay bounded.
(b) Coarse adaptive run of the actual problem: eta decreases monotonically
    and the final number of DOFs does not exceed max_dofs.
"""
import numpy as np
import pytest
from skfem import Functional
from skfem.helpers import dot, grad

import solver

PI = np.pi


def u_exact(x):
    return np.sin(PI * x[0]) * np.sin(PI * x[1]) * (1 + x[2])


def grad_u_exact(x):
    sx, cx = np.sin(PI * x[0]), np.cos(PI * x[0])
    sy, cy = np.sin(PI * x[1]), np.cos(PI * x[1])
    return np.array([PI * cx * sy * (1 + x[2]),
                     PI * sx * cy * (1 + x[2]),
                     sx * sy])


def lap_u_exact(x):
    return -2 * PI ** 2 * u_exact(x)


def f_exact(x):
    # -div(k grad u) = -(grad k . grad u + k lap u)
    return -(dot(solver.grad_k_coef(x), grad_u_exact(x))
             + solver.k_coef(x) * lap_u_exact(x))


def g_exact(x, n):
    return solver.k_coef(x) * dot(grad_u_exact(x), n)


MANUFACTURED = solver.Problem(
    k=solver.k_coef,
    grad_k=solver.grad_k_coef,
    f=f_exact,
    neumann={"front": g_exact, "back": g_exact},
    dirichlet=["left", "right", "bottom", "top"],
)


@Functional
def energy_error(w):
    e = grad_u_exact(w.x) - grad(w.uh)
    return solver.k_coef(w.x) * dot(e, e)


def test_manufactured_uniform():
    mesh = solver.build_mesh(3)
    ndofs, errs, etas = [], [], []
    for _ in range(3):
        basis, u = solver.assemble_and_solve(mesh, MANUFACTURED)
        eta = np.sqrt(np.sum(solver.estimate(basis, u, MANUFACTURED) ** 2))
        err = np.sqrt(energy_error.assemble(basis, uh=basis.interpolate(u)))
        ndofs.append(basis.N)
        errs.append(err)
        etas.append(eta)
        mesh = mesh.refined().with_defaults()
    ndofs, errs, etas = map(np.array, (ndofs, errs, etas))

    rates = -np.diff(np.log(errs)) / np.diff(np.log(ndofs))
    eff = etas / errs
    print("dofs", ndofs, "energy errors", errs, "rates", rates,
          "effectivity", eff)

    assert rates[-1] > 0.6          # optimal: 2/3 for P2 in 3D
    assert np.all((eff > 0.5) & (eff < 20.0))
    assert eff.max() / eff.min() < 2.0


def test_adaptive_coarse():
    max_dofs = 3000
    steps = solver.adaptive_loop(solver.build_mesh(3), theta=0.5,
                                 max_dofs=max_dofs, verbose=False)
    etas = np.array([s["eta"] for s in steps])
    ndofs = np.array([s["ndofs"] for s in steps])
    assert len(steps) >= 3
    assert ndofs[-1] <= max_dofs
    assert np.all(np.diff(ndofs) > 0)
    assert np.all(np.diff(etas) < 0)
    assert etas[-1] < 0.5 * etas[0]


def test_doerfler_marking():
    eta = np.array([4.0, 1.0, 3.0, 2.0])  # eta^2 = 16, 1, 9, 4 (sum 30)
    assert set(solver.doerfler_mark(eta, 0.5)) == {0}         # 16 >= 15
    assert set(solver.doerfler_mark(eta, 0.6)) == {0, 2}      # 25 >= 18


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
