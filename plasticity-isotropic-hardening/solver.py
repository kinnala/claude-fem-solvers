"""Elastoplastic tensile test of a flat steel specimen (plane stress, J2
plasticity with linear isotropic hardening), displacement controlled.

Strong form
-----------
Small strains, quasi-static, no body force, in-plane displacement u = (u_x, u_y):

    div sigma = 0                                in  Omega,
    eps = sym grad u = eps^e + eps^p,   sigma = C_ps : eps^e,
    f = sigma_vm - (sigma_y0 + H epsbar_p) <= 0,
    sigma_vm = sqrt(sxx^2 - sxx syy + syy^2 + 3 sxy^2),
    associative flow, Kuhn-Tucker loading/unloading conditions,

with C_ps the plane-stress elasticity tensor. Units: mm, N, MPa.
Quarter model of an ASTM E8 sheet-type dog-bone (x along the axis):

    sym_x (x = 0)     : u_x = 0
    sym_y (y = 0)     : u_y = 0
    grip  (x = 100)   : u_x = U(t), u_y free
    free  (the rest)  : traction-free

Weak form
---------
At load step n find u with the above Dirichlet data such that

    R(u; v) = int_Omega sigma(eps(u); history) : eps(v) dx = 0   for all v in V_0.

Newton: int_Omega eps(du) : C_alg : eps(v) dx = -R(u; v), where C_alg is the
consistent tangent of the plane-stress return map (Simo & Taylor; scalar
Newton for the plastic multiplier). sigma, eps^p and epsbar_p live at the
quadrature points. Each step starts from a tangent predictor; a backtracking
line search on the residual norm shortens Newton steps only when the full
step does not reduce the residual (e.g. when the whole reduced section
yields at once).

Outputs: reaction force F = 2 t sum(x-reactions on grip), extensometer
elongation dL = 2 u_x(25, 0), von Mises fields at max load and after unloading.
"""
import argparse
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from skfem import (  # noqa: E402
    Basis,
    BilinearForm,
    ElementTriP1,
    ElementTriP2,
    ElementVector,
    LinearForm,
    MeshTri,
    condense,
    solve,
)
from skfem.visuals.matplotlib import plot  # noqa: E402

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

PARAMS = {
    # material: mild steel
    "E": 210000.0,       # Young's modulus [MPa]
    "nu": 0.3,           # Poisson's ratio
    "sigma_y0": 250.0,   # initial yield stress [MPa]
    "H": 2000.0,         # linear isotropic hardening modulus [MPa]
    "t": 2.0,            # thickness [mm]
    # geometry (ASTM E8 sheet type, quarter model)
    "A": 57.0,           # length of reduced section [mm]
    "W": 12.5,           # width of reduced section [mm]
    "C": 20.0,           # width of grip section [mm]
    "R": 12.5,           # fillet radius [mm]
    "L": 200.0,          # overall length [mm]
    "gauge": 50.0,       # extensometer gauge length [mm]
    # mesh
    "h": 1.0,            # global mesh size [mm]
    "h_fillet": 0.5,     # mesh size in the fillets [mm]
    # loading (grip displacement on the quarter model)
    "u_max": 1.0,        # [mm], total elongation 2 u_max
    "n_load": 40,
    "n_unload": 20,
    # Newton
    "tol": 1e-8,
    "maxit": 25,
}

ELEMENT = ElementVector(ElementTriP2())
INTORDER = 2  # 3 quadrature points per triangle


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

def build_mesh(path="specimen.msh", h=PARAMS["h"], h_fillet=PARAMS["h_fillet"],
               p=PARAMS):
    """Quarter of the dog-bone in x in [0, L/2], y in [0, C/2] with Gmsh.
    Physical groups: sym_x, sym_y, grip, free (curves), specimen (surface)."""
    import gmsh

    a, w, c, R, L2 = p["A"] / 2, p["W"] / 2, p["C"] / 2, p["R"], p["L"] / 2
    # fillet: centre above the reduced-section edge, tangent at (a, w),
    # ends where it meets the grip width y = c
    yc = w + R
    xs = a + np.sqrt(R ** 2 - (yc - c) ** 2)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("specimen")
    geo = gmsh.model.geo
    pts = [geo.addPoint(*xy, 0.0) for xy in
           [(0, 0), (L2, 0), (L2, c), (xs, c), (a, w), (0, w)]]
    pc = geo.addPoint(a, yc, 0.0)
    l_symy = geo.addLine(pts[0], pts[1])
    l_grip = geo.addLine(pts[1], pts[2])
    l_top = geo.addLine(pts[2], pts[3])
    l_fil = geo.addCircleArc(pts[3], pc, pts[4])
    l_red = geo.addLine(pts[4], pts[5])
    l_symx = geo.addLine(pts[5], pts[0])
    loop = geo.addCurveLoop([l_symy, l_grip, l_top, l_fil, l_red, l_symx])
    surf = geo.addPlaneSurface([loop])
    geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [l_symx], name="sym_x")
    gmsh.model.addPhysicalGroup(1, [l_symy], name="sym_y")
    gmsh.model.addPhysicalGroup(1, [l_grip], name="grip")
    gmsh.model.addPhysicalGroup(1, [l_top, l_fil, l_red], name="free")
    gmsh.model.addPhysicalGroup(2, [surf], name="specimen")

    # local refinement around the fillet arc
    fd = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(fd, "CurvesList", [l_fil])
    gmsh.model.mesh.field.setNumber(fd, "Sampling", 200)
    ft = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(ft, "InField", fd)
    gmsh.model.mesh.field.setNumber(ft, "SizeMin", h_fillet)
    gmsh.model.mesh.field.setNumber(ft, "SizeMax", h)
    gmsh.model.mesh.field.setNumber(ft, "DistMin", 2.0)
    gmsh.model.mesh.field.setNumber(ft, "DistMax", 6.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(ft)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)

    gmsh.model.mesh.generate(2)
    gmsh.write(path)
    gmsh.finalize()
    return path


def load_mesh(path="specimen.msh"):
    mesh = MeshTri.load(path)
    for name in ["sym_x", "sym_y", "grip", "free"]:
        if name not in mesh.boundaries or len(mesh.boundaries[name]) == 0:
            raise RuntimeError(f"boundary '{name}' missing from {path}")
    return mesh


# ---------------------------------------------------------------------------
# Constitutive model: plane-stress J2 return map (Voigt, engineering shear)
# ---------------------------------------------------------------------------

def elastic_matrix(E, nu):
    return E / (1 - nu ** 2) * np.array([[1.0, nu, 0.0],
                                         [nu, 1.0, 0.0],
                                         [0.0, 0.0, (1 - nu) / 2]])


# projection with sigma^T P sigma = 2/3 sigma_vm^2
P_MAT = np.array([[2.0, -1.0, 0.0],
                  [-1.0, 2.0, 0.0],
                  [0.0, 0.0, 6.0]]) / 3.0


def von_mises(sig):
    sxx, syy, sxy = sig
    return np.sqrt(np.maximum(sxx ** 2 - sxx * syy + syy ** 2 + 3 * sxy ** 2,
                              0.0))


def return_map(eps, eps_p_n, alpha_n, mat):
    """Plane-stress J2 return map with linear isotropic hardening.

    eps, eps_p_n: (3, ...) Voigt strains (engineering shear),
    alpha_n: (...) equivalent plastic strain of the last converged step.
    Returns sigma (3, ...), eps_p (3, ...), alpha (...), tangent (3, 3, ...)."""
    E, nu, sy0, H = mat["E"], mat["nu"], mat["sigma_y0"], mat["H"]
    G = E / (2 * (1 + nu))
    Ce = elastic_matrix(E, nu)
    shape = alpha_n.shape

    sig_tr = np.einsum("ij,j...->i...", Ce, eps - eps_p_n)
    sigma = sig_tr.copy()
    eps_p = eps_p_n.copy()
    alpha = alpha_n.copy()
    D = np.broadcast_to(Ce[:, :, None], (3, 3, alpha_n.size)).reshape(
        (3, 3) + shape).copy()

    f_tr = von_mises(sig_tr) - (sy0 + H * alpha_n)
    pl = f_tr > 1e-10 * sy0
    if not np.any(pl):
        return sigma, eps_p, alpha, D

    s = sig_tr[:, pl]
    an = alpha_n[pl]
    a1 = (s[0] + s[1]) ** 2
    a2 = (s[1] - s[0]) ** 2
    a3 = s[2] ** 2
    k1 = E / (3 * (1 - nu))

    def xi_fun(dg):
        d1 = 1 + k1 * dg
        d2 = 1 + 2 * G * dg
        xi = a1 / (6 * d1 ** 2) + (0.5 * a2 + 2 * a3) / d2 ** 2
        dxi = -a1 * k1 / (3 * d1 ** 3) - (0.5 * a2 + 2 * a3) * 4 * G / d2 ** 3
        return xi, dxi

    # scalar Newton for the plastic multiplier dg at every plastic point:
    # psi(dg) = 1/2 xi(dg) - 1/3 sigma_y(alpha_n + dg sqrt(2/3 xi))^2 = 0
    dg = np.zeros_like(an)
    for _ in range(50):
        xi, dxi = xi_fun(dg)
        sq = np.sqrt(2.0 / 3.0 * xi)
        sy = sy0 + H * (an + dg * sq)
        psi = 0.5 * xi - sy ** 2 / 3.0
        dsy = H * (sq + dg * dxi / (3.0 * sq))
        dpsi = 0.5 * dxi - 2.0 / 3.0 * sy * dsy
        step = psi / dpsi
        dg = dg - step
        if np.all(np.abs(psi) <= 1e-13 * sy ** 2):
            break
    else:
        raise RuntimeError("return map did not converge")
    dg = np.maximum(dg, 0.0)

    xi, _ = xi_fun(dg)
    sy = sy0 + H * (an + dg * np.sqrt(2.0 / 3.0 * xi))
    # sigma = Xi C^{-1} sigma_trial, Xi = (C^{-1} + dg P)^{-1}
    Cinv = np.linalg.inv(Ce)
    M = Cinv[:, :, None] + dg[None, None, :] * P_MAT[:, :, None]
    Xi = np.linalg.inv(M.transpose(2, 0, 1))                  # (npl, 3, 3)
    sp = np.einsum("pij,jk,kp->ip", Xi, Cinv, s)
    n = P_MAT @ sp                                            # (3, npl)
    Xn = np.einsum("pij,jp->ip", Xi, n)
    beta = (4.0 / 9.0) * H * sy ** 2 / (1 - (2.0 / 3.0) * H * dg)
    denom = np.einsum("ip,ip->p", n, Xn) + beta
    Dp = Xi.transpose(1, 2, 0) - np.einsum("ip,jp->ijp", Xn, Xn) / denom

    sigma[:, pl] = sp
    eps_p[:, pl] = eps_p_n[:, pl] + dg * n
    alpha[pl] = an + dg * np.sqrt(2.0 / 3.0 * xi)
    D[:, :, pl] = Dp
    return sigma, eps_p, alpha, D


# ---------------------------------------------------------------------------
# Forms
# ---------------------------------------------------------------------------

def voigt(g):
    """Engineering Voigt strain [e_xx, e_yy, 2 e_xy] of a 2x2 gradient."""
    return np.array([g[0, 0], g[1, 1], g[0, 1] + g[1, 0]])


@BilinearForm
def tangent_form(du, v, w):
    return np.einsum("ij...,i...,j...->...", w.D, voigt(du.grad), voigt(v.grad))


@LinearForm
def internal_force(v, w):
    return np.einsum("i...,i...->...", w.sig, voigt(v.grad))


# ---------------------------------------------------------------------------
# Incremental solution
# ---------------------------------------------------------------------------

class Model:
    """Discretization, boundary DOFs and quadrature-point history."""

    def __init__(self, mesh, p=PARAMS, ext_x=None):
        self.p = p
        self.mesh = mesh
        self.basis = Basis(mesh, ELEMENT, intorder=INTORDER)
        nel, nq = mesh.nelements, self.basis.X.shape[1]
        self.grip_x = self.basis.get_dofs("grip").all("u^1")
        symx = self.basis.get_dofs("sym_x").all("u^1")
        symy = self.basis.get_dofs("sym_y").all("u^2")
        self.D = np.unique(np.concatenate([self.grip_x, symx, symy]))
        ext_x = p["gauge"] / 2 if ext_x is None else ext_x
        self.ext_x = ext_x
        self.probe = self.basis.probes(np.array([[ext_x], [0.0]]))
        # converged history
        self.u = self.basis.zeros()
        self.eps_p = np.zeros((3, nel, nq))
        self.alpha = np.zeros((nel, nq))
        self.sig = np.zeros((3, nel, nq))
        self.K = None
        self.f_ref = 0.0  # largest reaction norm of the converged history

    def elastic_stiffness(self):
        """Force F per unit grip displacement of the purely elastic body."""
        b = self.basis
        nel, nq = self.alpha.shape
        Ce = elastic_matrix(self.p["E"], self.p["nu"])
        D = np.broadcast_to(Ce[:, :, None, None], (3, 3, nel, nq))
        K = tangent_form.assemble(b, D=D)
        u = b.zeros()
        u[self.grip_x] = 1.0
        u = solve(*condense(K, b.zeros(), x=u, D=self.D))
        return self.reaction(K @ u)

    def state(self, u):
        eps = voigt(self.basis.interpolate(u).grad)
        return return_map(eps, self.eps_p, self.alpha, self.p)

    def reaction(self, fint):
        return 2.0 * self.p["t"] * np.sum(fint[self.grip_x])

    def elongation(self, u):
        return 2.0 * (self.probe @ u)[0]

    def solve_step(self, U, commit=True):
        """Newton solve for grip displacement U, starting from the last
        converged state. Returns dict with u, F, its, residual history."""
        b = self.basis
        I = b.complement_dofs(self.D)
        if self.K is None:
            sig, eps_p, alpha, D = self.state(self.u)
            self.K = tangent_form.assemble(b, D=D)
        # tangent predictor with the last converged tangent
        du = b.zeros()
        du[self.grip_x] = U - self.u[self.grip_x]
        du = solve(*condense(self.K, b.zeros(), x=du, D=self.D))
        u = self.u + du

        def evaluate(u):
            sig, eps_p, alpha, D = self.state(u)
            fint = internal_force.assemble(b, sig=sig)
            return (sig, eps_p, alpha, D), fint, np.linalg.norm(fint[I])

        res_hist = []
        st, fint, rnorm = evaluate(u)
        r0 = rnorm
        for it in range(self.p["maxit"] + 1):
            res_hist.append(rnorm)
            sig, eps_p, alpha, D = st
            ref = max(r0, np.linalg.norm(fint[self.D]), self.f_ref)
            K = tangent_form.assemble(b, D=D)
            if rnorm <= self.p["tol"] * ref:
                break
            if it == self.p["maxit"]:
                raise RuntimeError(f"Newton did not converge at U={U}")
            du = solve(*condense(K, -fint, D=self.D))
            # backtracking line search on the residual norm: the full
            # Newton step is taken whenever it reduces the residual
            # (always near the solution, so quadratic convergence is kept);
            # it is shortened when many points switch elastic/plastic at once
            s = 1.0
            for _ in range(12):
                st_new, f_new, r_new = evaluate(u + s * du)
                if r_new < (1 - 1e-4 * s) * rnorm:
                    break
                s *= 0.5
            u = u + s * du
            st, fint, rnorm = st_new, f_new, r_new
        out = {"u": u, "F": self.reaction(fint), "dL": self.elongation(u),
               "its": it, "res": res_hist, "sig": sig, "U": U}
        if commit:
            self.u, self.sig, self.eps_p, self.alpha, self.K = (
                u, sig, eps_p, alpha, K)
            self.f_ref = max(self.f_ref, np.linalg.norm(fint[self.D]))
        return out


def assemble_and_solve(mesh, p=PARAMS, ext_x=None, verbose=True):
    """Load to u_max in n_load steps, unload to F = 0 in n_unload steps.
    Returns (model, history list, snapshot at max load, final snapshot)."""
    model = Model(mesh, p, ext_x)
    k_el = model.elastic_stiffness()  # = elastic unloading stiffness
    hist = [{"U": 0.0, "dL": 0.0, "F": 0.0, "its": 0, "res": []}]

    def log(i, r, phase):
        hist.append(r)
        if verbose:
            print(f"{i:4d} {phase:6s} U={r['U']:9.5f} dL={r['dL']:9.5f} "
                  f"F={r['F'] / 1e3:9.4f} kN  its={r['its']}")

    if verbose:
        print("step phase  grip U [mm]  extens. dL [mm]  force F  Newton its")
    n_load = p["n_load"]
    for i in range(1, n_load + 1):
        log(i, model.solve_step(p["u_max"] * i / n_load), "load")
    max_snap = {"u": model.u.copy(), "sig": model.sig.copy(),
                "alpha": model.alpha.copy(), "F": hist[-1]["F"],
                "dL": hist[-1]["dL"]}
    F_max = max(h["F"] for h in hist)

    # unloading: elastic estimate of the zero-force grip position
    U_max, F_end = p["u_max"], hist[-1]["F"]
    U_star = U_max - F_end / k_el
    n_un = p["n_unload"]
    for i in range(1, n_un):
        log(n_load + i, model.solve_step(U_max - i / n_un * (U_max - U_star)),
            "unload")
    # last step: secant iteration on F(U) = 0 from the committed step n-1
    U0 = model.u[model.grip_x][0]
    F0 = hist[-1]["F"]
    U1 = U_star
    r = model.solve_step(U1, commit=False)
    for _ in range(30):
        if abs(r["F"]) < 1e-6 * F_max:
            break
        U0, F0, U1 = U1, r["F"], U1 - r["F"] * (U1 - U0) / (r["F"] - F0)
        r = model.solve_step(U1, commit=False)
    else:
        raise RuntimeError("secant iteration for F = 0 did not converge")
    r = model.solve_step(U1, commit=True)
    log(n_load + n_un, r, "unload")
    final_snap = {"u": model.u.copy(), "sig": model.sig.copy(),
                  "alpha": model.alpha.copy(), "F": r["F"], "dL": r["dL"]}
    return model, hist, max_snap, final_snap


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------

def project_p1(model, qp_values):
    pb = model.basis.with_element(ElementTriP1())
    return pb, pb.project(qp_values)


def plot_von_mises(model, sig, title, path):
    pb, vm = project_p1(model, von_mises(sig))
    fig, ax = plt.subplots(figsize=(11, 2.6))
    plot(pb, vm, ax=ax, shading="gouraud", colorbar=True, cmap="viridis")
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return vm


def postprocess(model, hist, max_snap, final_snap, outdir="."):
    os.makedirs(outdir, exist_ok=True)
    vm_max = plot_von_mises(
        model, max_snap["sig"],
        f"von Mises stress [MPa] at max load (F = {max_snap['F'] / 1e3:.2f} kN,"
        f" $\\Delta L$ = {max_snap['dL']:.3f} mm), quarter model",
        os.path.join(outdir, "von_mises_max_load.png"))
    vm_un = plot_von_mises(
        model, final_snap["sig"],
        f"residual von Mises stress [MPa] after unloading (F = 0, "
        f"permanent $\\Delta L$ = {final_snap['dL']:.3f} mm)",
        os.path.join(outdir, "von_mises_unloaded.png"))

    dL = np.array([h["dL"] for h in hist])
    F = np.array([h["F"] for h in hist]) / 1e3
    n = model.p["n_load"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(dL[:n + 1], F[:n + 1], "o-", ms=3, label="loading")
    ax.plot(dL[n:], F[n:], "s-", ms=3, label="unloading")
    ax.set_xlabel(r"extensometer elongation $\Delta L$ [mm] (gauge 50 mm)")
    ax.set_ylabel("force F [kN]")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title("load-displacement curve")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "load_displacement.png"), dpi=150)
    plt.close(fig)

    i_max = int(np.argmax(F))
    print(f"F_max            = {F[i_max]:.4f} kN at dL = {dL[i_max]:.5f} mm")
    print(f"permanent dL     = {final_snap['dL']:.5f} mm "
          f"(F = {final_snap['F']:.2e} N)")
    print(f"max von Mises    = {vm_max.max():.1f} MPa (max load), "
          f"{vm_un.max():.1f} MPa (residual)")
    print(f"max eps_p (eqv.) = {max_snap['alpha'].max():.5f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--h", type=float, default=PARAMS["h"],
                    help="global mesh size [mm] (default 1.0)")
    ap.add_argument("--h-fillet", type=float, default=PARAMS["h_fillet"],
                    help="mesh size in the fillets [mm] (default 0.5)")
    ap.add_argument("--n-load", type=int, default=PARAMS["n_load"],
                    help="number of loading steps (default 40)")
    ap.add_argument("--n-unload", type=int, default=PARAMS["n_unload"],
                    help="number of unloading steps (default 20)")
    ap.add_argument("--u-max", type=float, default=PARAMS["u_max"],
                    help="max grip displacement of the quarter model [mm] "
                         "(default 1.0, total elongation 2.0)")
    args = ap.parse_args(argv)
    logging.getLogger("skfem").setLevel(logging.ERROR)

    p = dict(PARAMS, h=args.h, h_fillet=args.h_fillet, n_load=args.n_load,
             n_unload=args.n_unload, u_max=args.u_max)
    outdir = os.path.dirname(os.path.abspath(__file__))
    msh = build_mesh(os.path.join(outdir, "specimen.msh"), p["h"],
                     p["h_fillet"], p)
    mesh = load_mesh(msh)
    print(f"mesh: {mesh.nelements} triangles, "
          f"{2 * (mesh.nvertices + mesh.nfacets)} DOFs")
    res = assemble_and_solve(mesh, p)
    postprocess(*res, outdir=outdir)
    return res


if __name__ == "__main__":
    main()
