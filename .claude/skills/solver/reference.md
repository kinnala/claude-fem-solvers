# scikit-fem / Gmsh reference for `/solver`

These snippets were checked against scikit-fem 12, gmsh 4.15 and meshio 5.3.
If the installed versions differ, check the API before relying on these.

## Gmsh mesh with named physical groups

Name every boundary segment and subdomain with a physical group. After
`MeshTri.load` / `MeshTet.load` (via meshio), the names show up as
`mesh.boundaries[name]` (facet indices) and `mesh.subdomains[name]`
(element indices). These names are the link to the boundary conditions.

```python
import gmsh

def build_mesh(path, h=0.05):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("domain")
    s = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
    gmsh.model.occ.synchronize()
    curves = [c[1] for c in gmsh.model.getBoundary([(2, s)], oriented=False)]
    # pick boundary curves geometrically, e.g. by center of mass
    left = [c for c in curves if abs(gmsh.model.occ.getCenterOfMass(1, c)[0]) < 1e-9]
    rest = [c for c in curves if c not in left]
    gmsh.model.addPhysicalGroup(1, left, name="left")
    gmsh.model.addPhysicalGroup(1, rest, name="rest")
    gmsh.model.addPhysicalGroup(2, [s], name="domain")
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    # gmsh.option.setNumber("Mesh.RecombineAll", 1)  # quads
    # gmsh.option.setNumber("Mesh.ElementOrder", 2)  # curved geometry -> MeshTri2
    gmsh.model.mesh.generate(2)
    gmsh.write(path)
    gmsh.finalize()
```

- Once there is a 2D physical group, Gmsh only writes the elements of the
  physical groups. Always add a physical group for the domain (or one per
  subdomain), or cells will be missing.
- In 3D, use `addBox`/`addCylinder`/boolean ops with surface (dim 2)
  boundary groups and volume (dim 3) groups, then `generate(3)`.
- After loading, print `mesh.boundaries.keys()` and check that every
  segment used in a BC is there and has a nonzero facet count. The
  segments together should cover `mesh.boundary_facets()`.
- If `import gmsh` fails with a missing `libGLU.so.1` (or `libXcursor`,
  `libXft`, `libXinerama`), install the system packages
  `libglu1-mesa libxcursor1 libxinerama1 libxft2 libxrender1`.

## Loading and bases

```python
from skfem import *
from skfem.helpers import dot, grad, div, ddot, sym_grad, trace, eye

mesh = MeshTri.load("mesh.msh")       # MeshQuad / MeshTet / MeshHex similarly
basis = Basis(mesh, ElementTriP2())
fbasis = FacetBasis(mesh, ElementTriP2(), facets=mesh.boundaries["rest"])
vbasis = Basis(mesh, ElementVector(ElementTriP2()))
pbasis = vbasis.with_element(ElementTriP1())  # same quadrature, other element
```

If a subdomain has its own coefficient, either pass
`elements=mesh.subdomains[name]` to `Basis`, or build a piecewise
coefficient array over elements and pass it as an extra field.

## Forms, assembly, BCs

```python
@BilinearForm
def a(u, v, w):
    return dot(grad(u), grad(v))

@BilinearForm
def robin(u, v, w):          # on a FacetBasis
    return 2.0 * u * v

@LinearForm
def f(v, w):
    x, y = w.x               # quadrature point coordinates
    return 1.0 * v

A = a.assemble(basis) + robin.assemble(fbasis)
b = f.assemble(basis)
D = basis.get_dofs("left")                 # dofs on a named boundary
x = basis.zeros()
x[D] = basis.project(lambda x: 0 * x[0])[D]  # or interpolate a Dirichlet value
u = solve(*condense(A, b, x=x, D=D))
```

- Neumann data: a `LinearForm` assembled over a `FacetBasis` (`w.n` is the
  outward normal).
- Vector problems: `vbasis.get_dofs("left").all()` constrains all
  components; `.all("u^1")` constrains only the first component (the names
  of the components depend on the element; check `get_dofs(...).nodal`).
- Non-homogeneous Dirichlet data: one simple way is
  `x = basis.project(g)`, which is accurate on nodal elements; for
  higher-order elements, use `basis.project(g, facets=...)` or interpolation
  through `FacetBasis` if needed.
- Extra fields: `a.assemble(basis, k=basis.interpolate(kvec))` makes `w.k`
  available in the form.
- Built-in models: `skfem.models.poisson.laplace, mass`;
  `skfem.models.elasticity.linear_elasticity, lame_parameters,
  linear_stress`.

## Linear elasticity

```python
from skfem.models.elasticity import linear_elasticity, lame_parameters, linear_stress
vb = Basis(mesh, ElementVector(ElementTriP1()))
lam, mu = lame_parameters(E, nu)            # plane strain; adjust for plane stress
K = linear_elasticity(lam, mu).assemble(vb)
u = solve(*condense(K, load.assemble(vb), D=vb.get_dofs("left").all()))
(ux, bx), (uy, by) = vb.split(u)            # component bases for plotting
```

## Mixed problems (block systems)

```python
from scipy.sparse import bmat
A = visc.assemble(vbasis)                   # ddot(grad(u), grad(v))
B = -divu.assemble(vbasis, pbasis)          # div(u) * q
K = bmat([[A, B.T], [B, None]], "csr")
```

Concatenate the dofs as `[velocity, pressure]`; the pressure dofs are
offset by `vbasis.N`. Fix the pressure constant if only velocity BCs are
given, for example by also constraining one pressure dof, or by adding a
small `eps * mass` block.

## Time-dependent problems

Assemble `M` (mass) and `A` once, then step. For theta-scheme with
`theta=1` (backward Euler) and `theta=0.5` (Crank–Nicolson):

```python
L = M + theta * dt * A
R = M - (1 - theta) * dt * A
for n in range(nsteps):
    u = solve(*condense(L, R @ u + dt * b, x=u_bc, D=D))
```

If the matrix does not change, factorize it once
(`scipy.sparse.linalg.splu` on the condensed matrix) and reuse the
factorization.

## Nonlinear problems (Newton)

```python
u = basis.zeros()          # set Dirichlet values in the initial guess
for it in range(maxit):
    uw = basis.interpolate(u)
    du = solve(*condense(J.assemble(basis, w=uw), -F.assemble(basis, w=uw), D=D))
    u += du
    if np.linalg.norm(du) < tol * max(1.0, np.linalg.norm(u)):
        break
else:
    raise RuntimeError("Newton did not converge")
```

Write `J` as the Gateaux derivative of `F` (for example
`(1 + w.w**2) * dot(grad(du), grad(v)) + 2 * w.w * du * dot(grad(w.w), grad(v))`).
With homogeneous Dirichlet data on `du`, the BCs stay satisfied.

## Eigenvalue problems

```python
from skfem.utils import solver_eigen_scipy_sym
L, X = solve(*condense(A, M, D=D), solver=solver_eigen_scipy_sym(k=6, sigma=0.0))
```

## Postprocessing

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skfem.visuals.matplotlib import plot, draw, plot3

ax = plot(basis, u, shading="gouraud", colorbar=True)  # P2 refined internally
draw(mesh, ax=ax)                                     # mesh overlay (optional)
ax.set_aspect("equal"); plt.savefig("u.png", dpi=150)
```

- Derived fields: `basis.project(basis.interpolate(u).grad[0])` for
  ∂u/∂x. For stresses, project componentwise into a scalar DG/P1 basis
  (for example `basis.with_element(ElementTriDG(ElementTriP1()))`).
- Scalar outputs: `@Functional def q(w): ...` and then
  `q.assemble(basis, u=basis.interpolate(u))`; on a `FacetBasis` for
  boundary fluxes (`dot(grad(w.u), w.n)`).
- Point values: `basis.probes(np.array([[x0], [y0]])) @ u`.
- Line plots: evaluate `basis.probes(points) @ u` along a sampled line and
  plot with `plt.plot`.
- 3D: matplotlib can only draw the boundary (`draw(mesh)`) or plot on a
  cut plane; for a full 3D view, also export to VTK:
  `mesh.save("u.vtk", point_data={"u": u[basis.nodal_dofs[0]]})`.
- Error against an exact solution:
  `@Functional def err(w): return (w.u - exact(w.x)) ** 2`, and
  `np.sqrt(err.assemble(basis, u=basis.interpolate(u)))`. A convergence
  study (h, h/2, h/4) with observed rates makes a good test.
