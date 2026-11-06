"""Convert a melt-pool mesh (STL or OBJ) into a penRed MESH_BODY .geo file with a
BSE scoring slab above it.

penRed frame used here: the beam travels along +z into the material. The highest
point of the surface (or `top_um`) sits at z = 0, the material at z > 0 and the BSE
slab at z < 0.
x and y are kept from the input mesh (after unit scaling and the up-axis rotation).

Bodies written, in this order (= IBODY index in penRed track output):
  0 meltpool      material `mat`      the mesh; an open surface is closed down to a flat base
  1 bse_detector  material `det_mat`  thin slab above the surface, detector 1
  2 world         material 0          vacuum box enclosing everything

The detector material must not be 0: penRed skips vacuum bodies, so a vacuum
detector never registers anything. Use the near-vacuum material (nearvac.mat).

The winding of the input does not matter as long as the mesh is manifold: triangles
are re-oriented so the outer surface faces out of the metal, and closed shells inside
the metal (pores) face into the pore, which penRed then treats as vacuum.

Small triangles: penRed's mesh tracker ignores a triangle when
|edge1 . (dir x edge2)| < 1e-8, an absolute tolerance in cm^2. Triangles of a few
um^2 are then missed by particles crossing at shallow angles, and sub-um triangles
are invisible altogether. The tool prints the expected share of missed crossings.
If it is not negligible, use `scale` (e.g. 1e4): all lengths are multiplied by
`scale`, and transport is unchanged as long as every material density is divided
by the same factor (only density x length enters the physics). With scale != 1:
  - create the materials at density / scale (pyPenred.simulation.materials.create)
  - multiply the source position and tally extents by scale (dsmax is done here)
  - divide PSF and track positions by scale; energies and directions are unchanged

Command line:
  python mesh2geo.py meltpool.stl -o meltpool.geo --units um
From Python (returns the geometry section for a pyPenred config dict):
  from mesh2geo import mesh_to_geo
  config["geometry"] = mesh_to_geo("meltpool.stl", "meltpool.geo", units="um")
  bodies = read_geo("meltpool.geo", scale=1.0)  # {name: (vertices, faces)}, cm, for plotting
Worked example: testing/meltpool/mesh2geo_example.ipynb
"""
import argparse
from pathlib import Path

import numpy as np

UNIT_TO_CM = {"m": 100.0, "cm": 1.0, "mm": 0.1, "um": 1e-4}
UM = 1e-4  # 1 um in cm
PENRED_DET_EPS = 1e-8  # penRed's triangle intersection tolerance, cm^2

# Proper rotations taking the given input axis to +z (face winding is unchanged)
UP_TO_Z = {
    "+z": np.eye(3),
    "-z": np.diag([1.0, -1.0, -1.0]),
    "+y": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float),
    "-y": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float),
    "+x": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float),
    "-x": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float),
}


# ---------------------------------------------------------------- reading

def read_mesh(path):
    """Return (vertices, faces) from an STL (ASCII or binary) or OBJ file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".stl":
        tris = _read_stl(path)
    elif suffix == ".obj":
        tris = _read_obj(path)
    else:
        raise ValueError(f"unsupported mesh format '{suffix}' (use .stl or .obj)")
    return _weld(tris)


def _read_stl(path):
    data = path.read_bytes()
    n = int.from_bytes(data[80:84], "little") if len(data) >= 84 else -1
    if len(data) == 84 + 50 * n:  # binary: the size matches the triangle count exactly
        rec = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        return np.frombuffer(data, dtype=rec, count=n, offset=84)["v"].astype(float)
    corners = [line.split()[1:4] for line in data.decode(errors="ignore").splitlines()
               if line.strip().startswith("vertex")]
    return np.array(corners, float).reshape(-1, 3, 3)


def _read_obj(path):
    verts, faces = [], []
    for line in path.read_text(errors="ignore").splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            verts.append([float(x) for x in p[1:4]])
        elif p[0] == "f":
            idx = [int(s.split("/")[0]) for s in p[1:]]
            idx = [i - 1 if i > 0 else len(verts) + i for i in idx]  # 1-based or negative
            faces += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]  # fan polygons
    return np.array(verts, float)[np.array(faces)]


def _weld(tris):
    """Merge duplicate corners of an (n, 3, 3) triangle soup into shared vertices."""
    pts = tris.reshape(-1, 3)
    tol = 1e-9 * np.ptp(pts, axis=0).max()
    _, first, inv = np.unique(np.round(pts / tol), axis=0, return_index=True, return_inverse=True)
    v, f = pts[first], inv.reshape(-1, 3)
    # drop degenerate triangles (repeated corners)
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    return v, f


# ---------------------------------------------------------------- topology

def _edge_keys(f, n):
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    return e, e[:, 0] * n + e[:, 1], e[:, 1] * n + e[:, 0]


def _boundary_edges(v, f):
    """Directed edges with no opposite edge. Raises on non-manifold or mis-oriented meshes."""
    e, key, rev = _edge_keys(f, len(v))
    if len(np.unique(key)) != len(key):
        raise ValueError("faces are not consistently oriented (or an edge is shared by more "
                         "than two faces); repair the mesh first, e.g. in MeshLab or Blender")
    return e[~np.isin(rev, key)]


def _boundary_loop(edges):
    nxt = dict(zip(edges[:, 0].tolist(), edges[:, 1].tolist()))
    loop = [int(edges[0, 0])]
    for _ in range(len(edges)):
        b = nxt.get(loop[-1])
        if b is None or b == loop[0]:
            break
        loop.append(b)
    if len(loop) != len(edges):
        raise ValueError("the open surface has holes or several separate edges; "
                         "it needs exactly one outer boundary to be closed into a solid")
    return np.array(loop)


def _signed_volume(v, f):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0


def _shells(f):
    """Make the winding consistent within each connected shell: neighbouring triangles
    must run along their shared edge in opposite directions.

    Returns (faces, shell index of each face, number of triangles flipped).
    """
    nf, n = len(f), f.max() + 1
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    face = np.tile(np.arange(nf), 3)
    key = e.min(axis=1) * n + e.max(axis=1)
    if np.unique(key, return_counts=True)[1].max() > 2:
        raise ValueError("an edge is shared by more than two faces; repair the mesh first, "
                         "e.g. in MeshLab or Blender")
    order = np.argsort(key, kind="stable")
    i = np.flatnonzero(key[order][1:] == key[order][:-1])
    a, b = order[i], order[i + 1]                       # the two uses of each shared edge
    same = (e[a, 0] < e[a, 1]) == (e[b, 0] < e[b, 1])   # same direction: one face must flip

    # Face adjacency in CSR form, then a flood fill per shell
    src, dst, rel = np.r_[face[a], face[b]], np.r_[face[b], face[a]], np.r_[same, same]
    o = np.argsort(src, kind="stable")
    start = np.searchsorted(src[o], np.arange(nf + 1)).tolist()
    dst, rel = dst[o].tolist(), rel[o].tolist()
    flip, shell, n_shells = [False] * nf, [-1] * nf, 0
    for seed in range(nf):
        if shell[seed] >= 0:
            continue
        shell[seed], stack = n_shells, [seed]
        while stack:
            p = stack.pop()
            for q, r in zip(dst[start[p]:start[p + 1]], rel[start[p]:start[p + 1]]):
                want = flip[p] != r
                if shell[q] < 0:
                    shell[q], flip[q] = n_shells, want
                    stack.append(q)
                elif flip[q] != want:
                    raise ValueError("the surface is not orientable; repair the mesh first")
        n_shells += 1

    flip, shell = np.array(flip), np.array(shell)
    flip ^= (np.bincount(shell, flip) > np.bincount(shell) / 2)[shell]  # flip the minority
    return np.where(flip[:, None], f[:, ::-1], f), shell, int(flip.sum())


def _ray_crossings(p, d, tri):
    """Number of triangles (n, 3, 3) that the ray p + t * d, t > 0, passes through."""
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    h = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, h)
    s = p - tri[:, 0]
    q = np.cross(s, e1)
    with np.errstate(divide="ignore", invalid="ignore"):  # zero-area triangles give NaN: no hit
        u = np.einsum("ij,ij->i", s, h) / det
        w = q @ d / det
        t = np.einsum("ij,ij->i", q, e2) / det
        return int(((u >= 0) & (w >= 0) & (u + w <= 1) & (t > 0)).sum())


def _orient_shells(v, f, shell):
    """Flip whole shells so that each faces away from the solid. A shell inside an odd number
    of other shells bounds a cavity (a pore) and has to face into it.

    Returns (faces, number of cavities).
    """
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    vol = np.bincount(shell, np.einsum("ij,ij->i", a, np.cross(b, c)))
    want = np.ones(len(vol))
    if len(vol) > 1:
        tri = v[f]
        d = np.array([0.137, 0.271, 0.953])  # off-axis, so the ray avoids grid-aligned edges
        for s in range(len(vol)):
            p = tri[np.argmax(shell == s)].T @ [0.2, 0.3, 0.5]  # a point on shell s
            if _ray_crossings(p, d, tri[shell != s]) % 2:
                want[s] = -1
    flip = (vol * want < 0)[shell]
    return np.where(flip[:, None], f[:, ::-1], f), int((want < 0).sum())


def _close_surface(v, f, depth):
    """Close an open, upward-facing surface into a solid: side walls down to a flat base."""
    loop = _boundary_loop(_boundary_edges(v, f))  # counter-clockwise seen from above
    n, m = len(v), len(loop)
    z_base = v[:, 2].min() - depth
    bottom = np.c_[v[loop, :2], np.full(m, z_base)]
    centre = np.r_[bottom[:, :2].mean(axis=0), z_base]

    # The base is a fan around the boundary centroid, which needs a star-shaped outline
    d1 = bottom[:, :2] - centre[:2]
    d2 = np.roll(d1, -1, axis=0)
    if (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0] <= 0).any():
        raise ValueError("the surface outline is not star-shaped around its centroid; "
                         "crop the mesh to a rectangle or circle first")

    a, b = loop, np.roll(loop, -1)               # top edge a -> b, as used by the surface
    a2 = n + np.arange(m)                        # matching bottom vertices
    b2 = np.roll(a2, -1)
    c = np.full(m, n + m)                        # base centre vertex
    walls = np.r_[np.c_[b, a, a2], np.c_[b, a2, b2]]
    base = np.c_[c, b2, a2]                      # facing -z
    return np.vstack([v, bottom, centre]), np.vstack([f, walls, base])


def _missed_crossings(v, f):
    """Area-weighted share of surface crossings that penRed's tracker will miss.

    A triangle is skipped when 2 * area * |cos| < PENRED_DET_EPS. For crossings of an
    isotropic flux |cos| is linearly distributed, so that happens with probability
    (PENRED_DET_EPS / (2 * area))^2.
    """
    area2 = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    miss = np.minimum(1.0, (PENRED_DET_EPS / area2) ** 2)
    return (miss * area2).sum() / area2.sum()


def _check_solid(v, f, name):
    if len(_boundary_edges(v, f)):
        raise ValueError(f"body '{name}' is not closed")
    if _signed_volume(v, f) <= 0:
        raise ValueError(f"body '{name}' has inward-facing triangles")


def _box(lo, hi):
    """Axis-aligned box as (vertices, faces) with outward normals."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return v, f


# ---------------------------------------------------------------- writing

def write_geo(path, bodies, source=""):
    """Write bodies [(name, parent, material, vertices, faces), ...] in penRed mesh format."""
    lines = [f"# penRed MESH_BODY geometry written by mesh2geo.py from {source}",
             "# Number of objects:", f" {len(bodies)}"]
    for name, parent, mat, v, f in bodies:
        used, f = np.unique(f, return_inverse=True)  # drop unused vertices
        v, f = v[used], f.reshape(-1, 3)
        lines += [f"# Object: {name}",
                  "#MAT      #NFACES     #NVERTEX     #NAME        #PARENT NAME",
                  f" {mat:03d}      {len(f):07d}     {len(v):08d}     {name}        {parent}",
                  "# VERTEX LIST", "# Index  (X Y Z)"]
        lines += [f"{i:04d} {x:+.8E} {y:+.8E} {z:+.8E}" for i, (x, y, z) in enumerate(v)]
        lines += ["# FACES(triangles)"]
        lines += [f" {a:03d} {b:03d} {c:03d}" for a, b, c in f]
        lines += ["#", "#"]
    Path(path).write_text("\n".join(lines) + "\n")


def read_geo(path, scale=1.0):
    """Read a .geo written by write_geo as {name: (vertices, faces)}, lengths divided by `scale`."""
    lines = [l for l in Path(path).read_text().splitlines() if l.strip() and not l.startswith("#")]
    bodies, i = {}, 1
    for _ in range(int(lines[0])):
        _, nf, nv, name = lines[i].split()[:4]
        nf, nv = int(nf), int(nv)
        v = np.array([l.split()[1:4] for l in lines[i + 1:i + 1 + nv]], float) / scale
        f = np.array([l.split() for l in lines[i + 1 + nv:i + 1 + nv + nf]], int)
        bodies[name] = (v, f)
        i += 1 + nv + nf
    return bodies


# ---------------------------------------------------------------- main entry

def mesh_to_geo(mesh_file, geo_file, units="um", up="+z", depth_um=50.0, gap_um=2.0,
                thickness_um=1.0, margin_um=0.0, pad_um=5.0, mat=1, det_mat=2,
                dsmax_um=1.0, scale=1.0, top_um=None, verbose=True):
    """Convert `mesh_file` to a penRed .geo with a BSE slab on top.

    units        unit of the input mesh coordinates: m, cm, mm or um
    up           input axis the free surface faces (towards the beam): +z, -z, +y, ...
    depth_um     depth of the solid below the lowest surface point (open surfaces only)
    gap_um       distance from z = 0 (the highest point, or top_um) to the bottom of the slab
    thickness_um slab thickness
    margin_um    slab overhang past the mesh footprint in x and y
    pad_um       vacuum between the bodies and the world box
    mat, det_mat material numbers of the melt pool and the detector slab
    dsmax_um     maximum step length inside the melt pool
    scale        multiply all lengths in the .geo by this (see the module notes)
    top_um       height in the mesh's own frame, in um, that becomes z = 0; default the
                 mesh's highest point. Give every frame of a time series the same value
                 (at least the highest point of any frame) so they share one frame.

    Returns the 'geometry' section for a pyPenred config dictionary.
    """
    v, f = read_mesh(mesh_file)
    v = v * UNIT_TO_CM[units] @ UP_TO_Z[up].T
    f, shell, n_fixed = _shells(f)
    n_surface = len(f)  # closing appends walls and a base after the input triangles

    if len(_boundary_edges(v, f)):  # open surface: make its normals face up, then close it
        normal_z = np.bincount(shell, np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])[:, 2])
        f = np.where((normal_z < 0)[shell][:, None], f[:, ::-1], f)
        v, f = _close_surface(v, f, depth_um * UM)
        f, shell, _ = _shells(f)  # the walls and base join the surface's shell
        closed_by_tool = True
    else:
        closed_by_tool = False
    f, n_cavities = _orient_shells(v, f, shell)

    # Into the penRed frame: depth = top - height (mirror z, so reverse the winding)
    top = v[:, 2].max() if top_um is None else top_um * UM
    if v[:, 2].max() > top:
        raise ValueError(f"the mesh reaches {v[:, 2].max() / UM:.2f} um, above top_um = {top_um} um")
    v = np.c_[v[:, :2], top - v[:, 2]]
    f = f[:, ::-1]

    lo, hi = v.min(axis=0), v.max(axis=0)
    m, gap, thk, pad = margin_um * UM, gap_um * UM, thickness_um * UM, pad_um * UM
    det = _box([lo[0] - m, lo[1] - m, -gap - thk], [hi[0] + m, hi[1] + m, -gap])
    world = _box([lo[0] - m - pad, lo[1] - m - pad, -gap - thk - pad],
                 [hi[0] + m + pad, hi[1] + m + pad, hi[2] + pad])

    bodies = [("meltpool", "world", mat, v * scale, f),
              ("bse_detector", "world", det_mat, det[0] * scale, det[1]),
              ("world", "void", 0, world[0] * scale, world[1])]
    for name, _, _, bv, bf in bodies:
        _check_solid(bv, bf, name)
    write_geo(geo_file, bodies, source=Path(mesh_file).name)
    missed = _missed_crossings(v * scale, f[:n_surface])

    if verbose:
        um = 1e4
        print(f"Wrote {geo_file}: {len(f)} triangles in the melt pool"
              f"{' (open surface, closed by mesh2geo)' if closed_by_tool else ''}")
        if n_fixed:
            print(f"  fixed the winding of {n_fixed} triangles that faced the wrong way")
        if shell.max() > 0:
            print(f"  {shell.max() + 1} separate shells, {n_cavities} of them pores inside the metal")
        print(f"  melt pool  x [{lo[0]*um:.2f}, {hi[0]*um:.2f}]  y [{lo[1]*um:.2f}, {hi[1]*um:.2f}]"
              f"  z [{lo[2]*um:.2f}, {hi[2]*um:.2f}] um  (z = depth below {top*um:.2f} um in the mesh's frame)")
        print(f"  BSE slab   z [{-(gap + thk)*um:.2f}, {-gap*um:.2f}] um")
        print(f"  Put the source between the slab and the surface (z between {-gap*um:.2f} and 0 um),"
              f" or above the slab (z between {-(gap + thk + pad)*um:.2f} and {-(gap + thk)*um:.2f} um).")
        if scale != 1:
            print(f"  Lengths in the file are x{scale:g} (the positions above are physical um):"
                  f" divide material densities by {scale:g}, multiply source and tally"
                  f" positions by {scale:g}, divide PSF/track positions by {scale:g}.")
        print(f"  penRed will miss about {missed:.2%} of surface crossings (triangle size vs its"
              f" 1e-8 cm^2 tolerance)")
        if missed > 1e-3:
            print("  WARNING: that is not negligible. Rerun with scale=1e4 (see the module notes).")

    return {"type": "MESH_BODY", "input-file": str(geo_file),
            "kdet": {"bse_detector": 1}, "dsmax": {"meltpool": dsmax_um * UM * scale}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mesh", help="input mesh (.stl or .obj)")
    ap.add_argument("-o", "--out", help="output .geo (default: input name with .geo)")
    ap.add_argument("--units", default="um", choices=UNIT_TO_CM, help="input mesh units (default um)")
    ap.add_argument("--up", default="+z", choices=UP_TO_Z, help="axis the surface faces (default +z)")
    ap.add_argument("--depth", type=float, default=50.0, help="solid depth below the surface for open meshes, um")
    ap.add_argument("--gap", type=float, default=2.0, help="surface top to slab bottom, um")
    ap.add_argument("--thickness", type=float, default=1.0, help="slab thickness, um")
    ap.add_argument("--margin", type=float, default=0.0, help="slab overhang past the mesh footprint, um")
    ap.add_argument("--pad", type=float, default=5.0, help="vacuum around the bodies inside the world, um")
    ap.add_argument("--mat", type=int, default=1, help="melt pool material number")
    ap.add_argument("--det-mat", type=int, default=2, help="detector slab material number (not 0)")
    ap.add_argument("--scale", type=float, default=1.0, help="length scale factor (see module notes)")
    args = ap.parse_args()

    out = args.out or str(Path(args.mesh).with_suffix(".geo"))
    geo = mesh_to_geo(args.mesh, out, args.units, args.up, args.depth, args.gap,
                      args.thickness, args.margin, args.pad, args.mat, args.det_mat,
                      scale=args.scale)
    print("\nConfig lines:")
    print(f'geometry/type "MESH_BODY"\ngeometry/input-file "{out}"')
    print(f"geometry/kdet/bse_detector 1\ngeometry/dsmax/meltpool {geo['dsmax']['meltpool']:.1e}")
