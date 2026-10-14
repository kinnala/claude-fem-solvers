# poisson-3d-adaptive

Adaptive P2 finite element solver for a variable-coefficient Poisson problem
in the unit cube. It uses a residual a posteriori error estimator, Dörfler
marking and scikit-fem's local refinement of tetrahedra.

## Specification

```
## Solver specification: poisson-3d-adaptive
1. Governing equation: -∇·(k ∇u) = f in Ω = (0,1)³, scalar unknown u.
   k(x,y,z) = x·y·z + 1, f = 0; nondimensional; stationary; linear.
   Weak form: find u ∈ V = {v ∈ H¹(Ω): v = 0 on Γ_D} such that
     ∫_Ω k ∇u·∇v dx = ∫_Ω f v dx + ∫_{Γ_N} g v ds   for all v ∈ V,
   with Γ_N = front ∪ back and g = k ∂u/∂n (outward normal).
2. Boundary conditions:
   front (z=0)  → Neumann, k ∂u/∂n = +1
   back  (z=1)  → Neumann, k ∂u/∂n = -1
   left (x=0), right (x=1), bottom (y=0), top (y=1) → Dirichlet u = 0
3. Elements: tetrahedra, ElementTetP2.
4. Geometry/mesh: unit cube built in scikit-fem (no Gmsh):
   MeshTet.init_tensor with 3 nodes per direction (48 tets, 125 P2 DOFs).
   Adaptive loop SOLVE → ESTIMATE → MARK → REFINE:
   - Residual estimator per element T:
     η_T² = h_T² ‖f + ∇·(k∇u_h)‖²_T
          + ½ Σ_{interior F⊂∂T} h_F ‖[[k ∇u_h·n]]‖²_F
          + Σ_{F⊂∂T∩Γ_N} h_F ‖g − k ∇u_h·n‖²_F,
     h_T, h_F = element/facet diameters; η = (Σ η_T²)^{1/2}
   - Dörfler marking, θ = 0.5
   - Refinement: mesh.refined(marked)
   - Stop: if refinement would produce more than 10 000 P2 DOFs (total, incl. Dirichlet dofs),
     discard that mesh and stop, so the final solved system has ≤ 10 000 DOFs.
   - CLI options: --theta (0.5), --max-dofs (10000), --n0 (3).
5. Postprocessing:
   - estimator_vs_dofs.png: η vs N_DOF, loglog, with an N^{-2/3} reference slope
   - final_mesh.png: boundary surface mesh of the final adapted mesh
   - VTK: solution_initial.vtk, solution_midway.vtk (step ⌊n_steps/2⌋), solution_final.vtk,
     each with u_h at vertices (point data) and η_T (cell data)
   - Printed table + convergence.csv: step, N_elements, N_DOF, η
   Verification (test_solver.py):
   (a) manufactured u* = sin(πx) sin(πy) (1 + z), same k; uniform refinement:
       energy error rate ≈ N^{-2/3}, effectivity index bounded across levels
   (b) coarse adaptive run (--max-dofs 3000): η decreases, final N_DOF ≤ max-dofs
```

### Implementation notes

- **Second derivatives:** `ElementTetP2` has no Hessian in scikit-fem.
  ∇u_h is piecewise linear, so its L² projection onto discontinuous P1 is
  exact. Differentiating that projection gives the exact Laplacian, which is
  used in the element residual ∇·(k∇u_h) = ∇k·∇u_h + k Δu_h.
- **Boundary tags:** in scikit-fem 12, adaptive `MeshTet.refined(marked)`
  does not carry the boundary tags over correctly. The solver therefore
  re-tags the cube faces with `.with_defaults()` after every refinement.
- **Midway step:** the midway snapshot is step `last // 2`, where `last`
  is the index of the final step.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python solver.py            # writes outputs into the current directory
pytest -q                   # verification (about 3 s)
```

### Command-line options

| option        | default | meaning                                      |
|---------------|---------|----------------------------------------------|
| `--n0`        | 3       | initial mesh: nodes per direction of the cube |
| `--theta`     | 0.5     | Dörfler bulk parameter                        |
| `--max-dofs`  | 10000   | maximum number of P2 DOFs in the final system |
| `--outdir`    | `.`     | output directory                              |

## Output files

| file                    | contents                                                  |
|-------------------------|-----------------------------------------------------------|
| `estimator_vs_dofs.png` | η against the number of DOFs (loglog) with an N^{-2/3} slope |
| `final_mesh.png`        | boundary surface of the final adapted mesh (two views)     |
| `solution_initial.vtk`  | initial mesh: `u` (point data), `eta` (cell data)          |
| `solution_midway.vtk`   | midway step: `u`, `eta`                                   |
| `solution_final.vtk`    | final mesh: `u`, `eta`                                    |
| `convergence.csv`       | step, elements, DOFs, η                                   |

Open the `.vtk` files in ParaView.
