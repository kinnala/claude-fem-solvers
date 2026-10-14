# claude-fem-solvers

Finite element solvers written in Python with [scikit-fem](https://github.com/kinnala/scikit-fem).
Each solver is in its own subdirectory.

## Creating a new solver

In Claude Code, run:

```
/solver <directory name>
```

The skill (`.claude/skills/solver/`) starts by interviewing you about the problem.
It does not write code until all of these are settled and you have confirmed them:

1. governing equation (strong and weak forms, coefficients, transient/nonlinear details)
2. boundary conditions on every boundary segment
3. element type(s)
4. geometry and mesh generation (Gmsh by default)
5. postprocessing (matplotlib through `skfem.visuals.matplotlib` by default) and verification

It then writes `solver.py`, `README.md`, `requirements.txt`, `test_solver.py` and
`.gitignore` in the new directory, runs the solver and the tests, and checks the
generated figures.
