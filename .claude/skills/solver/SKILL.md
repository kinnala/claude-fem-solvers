---
name: solver
description: Create a new finite element solver (Python + scikit-fem) in a new subdirectory of the repository root. Starts with a mandatory interview that pins down the governing equation, boundary conditions, element type, geometry/mesh and postprocessing before any code is written. Use when the user runs /solver <directory name> or asks to create a new FEM solver.
argument-hint: <directory name>
disable-model-invocation: true
---

# /solver — create a new scikit-fem solver

Target directory: `$ARGUMENTS` (relative to the repository root).

The work has three phases. **Never skip or reorder them.** Do not write any
files in the target directory until Phase 1 has ended with the user's
explicit confirmation.

## Phase 0 — Preflight

1. If `$ARGUMENTS` is empty, ask the user for the directory name and stop
   until you have one. The name must be one path segment
   (`[A-Za-z0-9_-]+`), with no `/`, `..` or spaces. Suggest a sanitized name
   if needed.
2. Resolve it against the repository root (`git rev-parse --show-toplevel`).
   If the directory already exists and is not empty, show the user what is
   there and ask whether to pick another name or stop. Never overwrite an
   existing solver.
3. Look at the existing solver directories in the root (if any) and follow
   their layout and conventions, so the repository stays consistent.

## Phase 1 — Interview (mandatory, blocking)

Find out what problem the user wants solved. Start by asking them to
describe it in their own words, unless they already did so in the message
that invoked the skill. Then work through the checklist below until
**every** item is settled. Use `AskUserQuestion` for questions that have
discrete options; use plain questions for open-ended ones (formulas,
dimensions, numbers). Ask a few related questions per round rather than one
at a time or all at once.

Do not finish the interview based on your own assumptions. When an item has
a sensible default, **propose** it and have the user accept it. A default
that has not been confirmed counts as an open item.

### Checklist

**1. Governing equation**
- The PDE in strong form, with every unknown named (scalar or vector field,
  how many components, any coupled/mixed fields).
- Stationary or time-dependent. If time-dependent: the time interval, time
  step, time-stepping scheme (for example backward Euler, Crank–Nicolson)
  and the initial condition.
- Linear or nonlinear. If nonlinear: the linearization (Picard or Newton),
  tolerance and maximum number of iterations.
- Every coefficient and source term with its value or formula (constant,
  piecewise per subdomain, or a function of x/t/u), plus units or a
  statement that the problem is nondimensional.
- Eigenvalue problems: how many eigenpairs are wanted and which part of the
  spectrum.
- Write out the weak (variational) form you plan to implement and have the
  user confirm it. Most misunderstandings show up here.

**2. Boundary conditions**
- Every part of the boundary is covered, with no gaps or overlaps: for each
  named boundary segment, the type (Dirichlet / Neumann / Robin / periodic /
  symmetry / traction / slip, …) and the prescribed value or formula.
- For vector problems: which components are constrained on each segment.
- If the problem is only determined up to a constant (pure Neumann, pressure
  in enclosed flow, floating elastic body): how the solution is made unique
  (fixed point, zero mean, Lagrange multiplier, …).
- Interface conditions between subdomains, if any.

**3. Element type**
- The cell shape (triangle/quadrilateral in 2D, tetrahedron/hexahedron in
  3D) and the scikit-fem element class(es) for each field, for example
  `ElementTriP1`, `ElementTriP2`, `ElementQuad2`, `ElementTetP2`,
  `ElementVector(ElementTriP2())`, or a Taylor–Hood pair
  `ElementTriP2`/`ElementTriP1` for Stokes.
- For mixed problems, check that the pair is inf-sup stable and tell the
  user if it is not.
- If the user has no preference, propose a standard choice with a one-line
  reason and get it confirmed.

**4. Geometry and mesh generation**
- The spatial dimension and an exact description of the geometry (shapes,
  dimensions, holes, subdomains), or a CAD/mesh file to import.
- The default is **Gmsh** through its Python API (`gmsh` package). Use
  another approach only if the user asks for it (for example a built-in
  scikit-fem mesh such as `MeshTri.init_sqsymmetric()` for a unit square, or
  importing an existing mesh with meshio).
- A target mesh size (global and any local refinement) and whether the mesh
  size should be a command-line parameter.
- The names of the physical groups for each boundary segment and subdomain.
  These names link item 2 to the mesh, so use them consistently.

**5. Postprocessing**
- Which quantities are wanted: the solution field(s), derived quantities
  (gradient, flux, stress, von Mises, vorticity, …), scalar outputs
  (integrals, point values, reaction forces, eigenvalues) and line plots.
- How results are delivered. The default is **matplotlib**, through the
  scikit-fem plotting interface (`skfem.visuals.matplotlib`: `plot`, `draw`,
  `plot3`), saved as PNG files in the solver directory. Also ask whether
  values should be printed or saved (CSV/NPZ), whether a VTK/XDMF export
  for ParaView is wanted (`mesh.save(..., point_data=...)`), and whether
  time-dependent problems need snapshots or an animation.
- Verification: whether there is an analytical/manufactured solution or a
  reference value the solver should be checked against. Propose one when
  it is cheap to do (for example a manufactured solution on a simple
  geometry).

### Exit criterion

The interview ends only when all of the following are true:

1. Every checklist item has a concrete answer. Nothing is left as "TBD",
   "something reasonable" or a default the user has not confirmed.
2. The answers are consistent and determine the solver uniquely: the
   problem is well-posed with these BCs, the elements suit the equation,
   every boundary condition refers to a physical group in the mesh, and
   every requested output can be computed from the solution. If anything
   conflicts or is ambiguous, raise it and resolve it with the user.
3. You have shown the user a **specification summary** in this format and
   they have explicitly confirmed it (for example "yes", "confirmed",
   "go ahead"):

```
## Solver specification: <directory name>
1. Governing equation: <strong form>; weak form: <weak form>;
   coefficients: <...>; stationary/transient: <...>; linear/nonlinear: <...>
2. Boundary conditions: <segment name → type, value> (all segments listed)
3. Elements: <cell type, scikit-fem element per field>
4. Geometry/mesh: <geometry, dimensions, Gmsh, mesh size, physical groups>
5. Postprocessing: <plots, derived quantities, outputs, file names>
   Verification: <check that will be run>
```

If the user changes anything in reply to the summary, update it and ask
for confirmation again. Only a confirmation of the latest summary ends
Phase 1.

## Phase 2 — Implementation

Read [reference.md](reference.md) for the scikit-fem/Gmsh patterns and
API notes before writing code.

Create `<repo root>/$ARGUMENTS/` with the following files:

- `solver.py`: a single runnable script with these sections, in order:
  parameters (a top-level dict or dataclass plus `argparse` for the
  important ones, mesh size at least), `build_mesh()` (Gmsh, writes `.msh`),
  `load_mesh()`, bilinear/linear forms, `assemble_and_solve()`,
  `postprocess()`, and `main()` guarded by `if __name__ == "__main__":`.
  Comment the physics briefly (the strong and weak forms in the module
  docstring). Do not use interactive windows: use `matplotlib.use("Agg")`
  and save figures to files.
- `README.md`: the confirmed specification summary, how to install and
  run, the command-line options, and a list of the output files.
- `requirements.txt`: `scikit-fem`, `numpy`, `scipy`, `matplotlib`,
  `meshio`, `gmsh`, plus anything else the solver uses.
- `test_solver.py`: a fast pytest check on a coarse mesh that runs the
  verification agreed in the interview (convergence/error against the
  manufactured or analytical solution, or comparison with the reference
  value).
- `.gitignore`: generated files (`*.msh`, `*.png`, `*.vtk`, `*.xdmf`,
  `*.h5`, `__pycache__/`), unless the user asked for outputs to be
  committed.

Implement exactly what was confirmed. If you find during implementation
that part of the specification cannot be done as agreed, stop and ask
the user. Do not change the specification on your own.

## Phase 3 — Verification and report

1. Install the requirements into a virtual environment and run
   `python solver.py` from the solver directory. If `import gmsh` fails
   with a missing `libGLU.so.1` / `libXcursor.so.1` / `libXft.so.2` /
   `libXinerama.so.1`, install the system libraries
   (`libglu1-mesa libxcursor1 libxinerama1 libxft2 libxrender1` on
   Debian/Ubuntu) and try again.
2. Run `pytest` in the solver directory. Both runs must succeed.
3. Look at every generated figure (read the PNGs) and check that it shows
   what the specification asked for and that the solution looks physically
   reasonable (signs, magnitudes, BCs visibly satisfied, no artifacts).
   Fix any problems.
4. Report back to the user: the files created, the command to run, the
   verification result (errors/rates or reference comparison, with
   numbers), key scalar outputs, and the generated figures. Commit only
   if the user asks you to, or if the session's instructions say to
   commit and push.
