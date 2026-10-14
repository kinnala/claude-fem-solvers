# plasticity-isotropic-hardening

Displacement-controlled tensile test of a flat steel specimen (ASTM E8 sheet
type) under plane stress. The material model is J2 (von Mises) plasticity
with linear isotropic hardening. The test loads the specimen to 2 mm total
elongation and then unloads it to zero force. The solver is Newton with the
consistent tangent of the plane-stress return map, and the plastic history
is stored at the quadrature points.

## Specification

```
## Solver specification: plasticity-isotropic-hardening
1. Governing equation: small-strain, rate-independent J2 (von Mises) elastoplasticity
   with linear isotropic hardening, plane stress, quasi-static.
   Unknown: in-plane displacement u = (u_x, u_y) [mm].
   Strong form: div σ = 0 in Ω (no body force), ε = sym ∇u = ε^e + ε^p, σ = C_ps : ε^e
     (C_ps = plane-stress elasticity),
     yield f = σ_vm − (σ_y0 + H ε̄p) ≤ 0, σ_vm = sqrt(σxx² − σxx σyy + σyy² + 3 σxy²),
     associative flow, KKT loading/unloading conditions.
   Coefficients: E = 210 000 MPa, ν = 0.3, σ_y0 = 250 MPa, H = 2 000 MPa, thickness t = 2 mm.
     Units: mm, N, MPa.
   Weak form (at load step n): find u ∈ V_U with u_x = U_n on grip, u_x = 0 on sym_x, u_y = 0 on sym_y:
     R(u; v) = ∫_Ω σ(ε(u); history) : ε(v) dx = 0   ∀ v ∈ V_0.
     Newton step: ∫_Ω ε(δu) : C_alg : ε(v) dx = −R(u; v), C_alg = consistent tangent
     of the plane-stress return map (Simo–Taylor; scalar Newton solve for Δγ), with
     σ, ε^p, ε̄p stored at the quadrature points.
   Transient? Quasi-static, incremental, displacement-controlled. Nonlinear: Newton,
     tolerance ‖R‖ ≤ 1e-8·max(‖R_0‖, ‖f_reaction‖), at most 25 iterations per step.
   Load history (U = grip displacement on quarter model):
     loading U: 0 → 1.0 mm in 40 equal steps (total elongation 2 mm);
     unloading: 20 steps towards the grip position U* where F = 0. U* is estimated from the
     elastic unloading stiffness, and the last step uses a secant iteration until
     |F| < 1e-6·F_max.
2. Boundary conditions (quarter model, x along the axis, y across the width, origin at the centre):
   sym_x (x = 0)          → u_x = 0, traction-free in y
   sym_y (y = 0)          → u_y = 0, traction-free in x
   grip  (end edge x=100) → u_x = U(t), u_y free
   free  (all other edges: specimen side, fillet, shoulder, grip side) → traction-free
   Force: F = 2 · t · Σ (x-reactions on grip DOFs) [N].
   Extensometer: ΔL = 2 · u_x(x = 25, y = 0) (gauge length 50 mm).
3. Elements: triangles, ElementVector(ElementTriP2()); quadrature order 2 (3 points per element)
   for the internal variables.
4. Geometry/mesh: ASTM E8 sheet-type dog-bone, quarter model in x ∈ [0, 100], y ∈ [0, 10]:
   reduced section half-length 28.5 mm (A = 57), half-width 6.25 mm (W = 12.5),
   fillet R = 12.5 mm tangent to the reduced section and ending at the shoulder of the grip
   section (half-width 10 mm, i.e. C = 20) that extends to x = 100 (overall 200 mm).
   Gmsh Python API → specimen.msh; h = 1.0 mm globally, 0.5 mm in the fillets.
   CLI: --h (1.0), --h-fillet (0.5), --n-load (40), --n-unload (20), --u-max (1.0).
   Physical groups: curves "sym_x", "sym_y", "grip", "free"; surface "specimen".
5. Postprocessing (matplotlib, skfem.visuals.matplotlib, PNG files in the solver directory):
   - von_mises_max_load.png: von Mises stress at max load (L² projected to P1, quarter model)
   - von_mises_unloaded.png: residual von Mises stress after unloading
   - load_displacement.png: F [kN] vs extensometer ΔL [mm], loading and unloading
   - stdout: table per step (step, U, ΔL, F, Newton its) + F_max, ΔL at F_max, permanent ΔL
   Verification (test_solver.py):
   (a) Rectangular strip (same BCs, plane stress → homogeneous uniaxial stress), loaded and
       unloaded: σ_xx and F compared with the exact bilinear law
       (σ = Eε, then σ_y0 + E_t(ε − ε_y), E_t = EH/(E+H); elastic unloading; permanent
       strain), rel. error < 1e-6; Newton convergence quadratic.
   (b) Coarse dog-bone run: initial slope F/ΔL matches the elastic estimate E·A_gauge/50,
       F_max matches the gauge-section analytic value (σ from ΔL/50, ×12.5×2) within 2%,
       permanent ΔL > 0.
```

Small-strain theory: the model does not capture necking.

### Implementation notes

- **Return map:** this follows Simo & Hughes (*Computational Inelasticity*,
  §3.4). Strains are in Voigt notation with engineering shear, and
  `P = 1/3 [[2,-1,0],[-1,2,0],[0,0,6]]` (σᵀPσ = ⅔σ_vm²). The plastic
  multiplier Δγ comes from a scalar Newton solve of
  ψ(Δγ) = ½ξ(Δγ) − ⅓σ_y² = 0. The consistent tangent is
  `Ξ − (Ξn)(Ξn)ᵀ / (nᵀΞn + β)` with `n = Pσ` and
  `β = (4/9) H σ_y² / (1 − ⅔HΔγ)`. Its central finite-difference error
  is about 1e-10.
- **Newton globalization:**
  - Each load step starts from a tangent predictor based on the last
    converged tangent.
  - Almost the whole reduced section yields in the same step. Full
    Newton steps diverge there, so a backtracking line search on the
    residual norm halves the step until the residual decreases. Near
    the solution the full step is always accepted, which keeps the
    convergence quadratic.
- **Tolerance reference:** after unloading to F = 0 the current reaction
  norm is also about zero. The `‖f_reaction‖` in the stopping test is
  therefore the largest reaction norm of the converged history, so the
  relative tolerance stays meaningful.
- **Zero-force grip position:** the solver estimates U* from the elastic
  stiffness of the structure. That stiffness is computed by an elastic
  solve with a unit grip displacement, and it equals the unloading
  stiffness.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python solver.py            # about 25 s, writes outputs into this directory
pytest -q                   # verification (about 2 s)
```

If `import gmsh` fails with a missing `libGLU.so.1` (or `libXcursor`, `libXft`,
`libXinerama`), install `libglu1-mesa libxcursor1 libxinerama1 libxft2 libxrender1`.

### Command-line options

| option        | default | meaning                                                    |
|---------------|---------|------------------------------------------------------------|
| `--h`         | 1.0     | global mesh size [mm]                                      |
| `--h-fillet`  | 0.5     | mesh size in the fillets [mm]                              |
| `--n-load`    | 40      | number of loading steps                                    |
| `--n-unload`  | 20      | number of unloading steps                                  |
| `--u-max`     | 1.0     | max grip displacement of the quarter model [mm] (total elongation 2·u_max) |

## Output files

| file                     | contents                                                   |
|--------------------------|------------------------------------------------------------|
| `specimen.msh`           | Gmsh mesh of the quarter specimen                          |
| `von_mises_max_load.png` | von Mises stress at max load                               |
| `von_mises_unloaded.png` | residual von Mises stress after unloading to F = 0         |
| `load_displacement.png`  | force F [kN] vs extensometer elongation ΔL [mm]            |

The per-step table and the summary (F_max, ΔL at F_max, permanent ΔL,
maximum von Mises stress, maximum equivalent plastic strain) are printed to
stdout.

### Reference results (default settings)

| quantity                    | value                           |
|-----------------------------|---------------------------------|
| mesh                        | 2502 P2 triangles, 10 482 DOFs  |
| F_max                       | 7.689 kN at ΔL = 1.527 mm       |
| permanent ΔL                | 1.454 mm                        |
| max von Mises (max load)    | 312 MPa                         |
| max residual von Mises      | 93 MPa                          |
| max equivalent plastic strain | 0.031                         |
