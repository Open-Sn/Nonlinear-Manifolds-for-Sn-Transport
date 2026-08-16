# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 22:56:34 2026

@author: jean.ragusa
"""

"""
Compute DG(P1) consistent mass matrices on a triangular VTU mesh, and build an
affine decomposition by material ID (CellData array, default "Block").

Mass on triangle K (area A_K):
  M_K = (A_K/12) * [[2,1,1],[1,2,1],[1,1,2]]

Because your DG mesh duplicates vertices, global M is 3x3 block diagonal (one block per cell).
We still assemble robustly from connectivity.

Requires: VTK python (pip install vtk)
Optional: SciPy for sparse save_npz (pip install scipy)
"""


# ----------------------------
# User settings
# ----------------------------
import os
import numpy as np
VTU_PATH = r"./aflux_3newss_1000_0.vtu"   # can be full path
CELL_MAT_NAME = "Block"            # CellData array with material IDs
OUT_DIR = "dg_mass_out"

DTYPE_DATA = np.float64            # accumulation dtype
SAVE_SPARSE = True                 # if SciPy is available
SAVE_COMPACT = True                # saves cell areas/material/connectivity/scales

# Plot settings
PLOT_AFTER_COMPUTE = True  # set False if you only want to compute/save
PLOT_PATTERN = "mass_mat_*.npz"
PLOT_SAVE_DIRNAME = "spy_plots"

PLOT_MARKERSIZE = 0.25
PLOT_DPI = 200
PLOT_SHOW = True  # set True for interactive popups

# ----------------------------
# VTK imports
# ----------------------------
try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except Exception as e:
    raise RuntimeError("Could not import VTK. Install with: pip install vtk\nError: {}".format(e))

# ----------------------------
# Optional SciPy sparse
# ----------------------------
try:
    if SAVE_SPARSE:
        import scipy.sparse as sp
        from scipy.sparse import save_npz
    else:
        sp = None
        save_npz = None
except Exception:
    sp = None
    save_npz = None


def ensure_dir(d):
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def read_vtu(path):
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(path)
    r.Update()
    return r.GetOutput()


def get_cell_connectivity_triangles(grid):
    """
    Return cell2pt as (ncells,3) int array for triangle cells.
    Assumes all cells are triangles (as in your file).
    """
    nc = grid.GetNumberOfCells()
    cell2pt = np.empty((nc, 3), dtype=np.int64)

    for c in range(nc):
        cell = grid.GetCell(c)
        if cell.GetNumberOfPoints() != 3:
            raise RuntimeError("Cell {} has {} points, expected triangles".format(
                c, cell.GetNumberOfPoints()))
        pid = cell.GetPointIds()
        cell2pt[c, 0] = pid.GetId(0)
        cell2pt[c, 1] = pid.GetId(1)
        cell2pt[c, 2] = pid.GetId(2)

    return cell2pt


def triangle_areas(points_xyz, cell2pt):
    """
    points_xyz: (npts,3)
    cell2pt: (ncells,3)
    area = 0.5 * || (p1-p0) x (p2-p0) ||
    """
    p0 = points_xyz[cell2pt[:, 0], :]
    p1 = points_xyz[cell2pt[:, 1], :]
    p2 = points_xyz[cell2pt[:, 2], :]

    v1 = p1 - p0
    v2 = p2 - p0
    cr = np.cross(v1, v2)
    A = 0.5 * np.linalg.norm(cr, axis=1)
    return A


def assemble_by_material(npts, cell2pt, areas, mats):
    """
    Build per-material COO triplets (rows, cols, data) for global DG mass matrices.
    Because DG vertices are duplicated, blocks do not overlap, but we do general assembly anyway.
    """
    # Template (without area scaling): [[2,1,1],[1,2,1],[1,1,2]]
    T = np.array([[2.0, 1.0, 1.0],
                  [1.0, 2.0, 1.0],
                  [1.0, 1.0, 2.0]], dtype=DTYPE_DATA)

    # Precompute local row/col pattern for a 3-node element
    rr = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
    cc = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
    tflat = T.reshape(-1)  # length 9

    mats_unique = np.unique(mats)
    acc = {}
    for m in mats_unique:
        acc[int(m)] = {"rows": [], "cols": [], "data": []}

    for c in range(cell2pt.shape[0]):
        m = int(mats[c])
        ids = cell2pt[c]  # length 3 point ids
        A = areas[c]
        scale = A / 12.0

        # Map local (0,1,2) indices to global point ids
        rows = ids[rr]
        cols = ids[cc]
        data = scale * tflat

        acc[m]["rows"].append(rows)
        acc[m]["cols"].append(cols)
        acc[m]["data"].append(data)

    # Concatenate per material
    for m in acc:
        acc[m]["rows"] = np.concatenate(acc[m]["rows"]).astype(np.int64, copy=False)
        acc[m]["cols"] = np.concatenate(acc[m]["cols"]).astype(np.int64, copy=False)
        acc[m]["data"] = np.concatenate(acc[m]["data"]).astype(DTYPE_DATA, copy=False)

    return acc


def main():
    ensure_dir(OUT_DIR)

    grid = read_vtu(VTU_PATH)

    npts = grid.GetNumberOfPoints()
    ncells = grid.GetNumberOfCells()
    print("VTU:", VTU_PATH)
    print("NumberOfPoints =", npts)
    print("NumberOfCells  =", ncells)

    # Points
    pts = vtk_to_numpy(grid.GetPoints().GetData()).astype(DTYPE_DATA, copy=False)  # (npts,3)

    # Connectivity (triangles)
    cell2pt = get_cell_connectivity_triangles(grid)

    # Material IDs (CellData)
    cd = grid.GetCellData()
    mat_arr = cd.GetArray(CELL_MAT_NAME)
    if mat_arr is None:
        names = [cd.GetArrayName(i) for i in range(cd.GetNumberOfArrays())]
        raise RuntimeError(
            "CellData array '{}' not found. Available: {}".format(CELL_MAT_NAME, names))
    mats = vtk_to_numpy(mat_arr).astype(np.int64, copy=False)

    # Areas
    areas = triangle_areas(pts, cell2pt)
    print("Area stats: min {:.6e}, max {:.6e}, mean {:.6e}".format(
        areas.min(), areas.max(), areas.mean()))
    print("Unique material IDs:", np.unique(mats))

    # Quick check: DG duplication (often npts == 3*ncells)
    if npts == 3 * ncells:
        print("Note: npts == 3*ncells, consistent with DG-duplicated vertices and 3x3 block diagonal mass.")
    else:
        print("Warning: npts != 3*ncells; blocks may overlap (still assembled correctly).")

    # Compact “per cell” scaling (useful regardless of material): M_cell = (A/12) * T
    mass_scale = areas / 12.0  # length ncells

    # Assemble affine decomposition by material
    acc = assemble_by_material(npts, cell2pt, areas, mats)

    # Save compact arrays (fast rebuild later)
    if SAVE_COMPACT:
        np.save(os.path.join(OUT_DIR, "cell2pt.npy"), cell2pt)
        np.save(os.path.join(OUT_DIR, "cell_areas.npy"), areas)
        np.save(os.path.join(OUT_DIR, "cell_material.npy"), mats)
        np.save(os.path.join(OUT_DIR, "mass_scale_A_over_12.npy"), mass_scale)
        T = np.array([[2.0, 1.0, 1.0],
                      [1.0, 2.0, 1.0],
                      [1.0, 1.0, 2.0]], dtype=DTYPE_DATA)
        np.save(os.path.join(OUT_DIR, "mass_template_T.npy"), T)
        print("Saved compact arrays to:", OUT_DIR)

    # Save per-material sparse matrices (and global) if SciPy available
    if sp is not None and save_npz is not None:
        M_global = None

        for m in sorted(acc.keys()):
            rows = acc[m]["rows"]
            cols = acc[m]["cols"]
            data = acc[m]["data"]

            M_m = sp.coo_matrix((data, (rows, cols)), shape=(npts, npts)).tocsc()
            out_m = os.path.join(OUT_DIR, "mass_mat_{}.npz".format(m))
            save_npz(out_m, M_m)
            print("Saved:", out_m, " nnz=", M_m.nnz)

            if M_global is None:
                M_global = M_m.copy()
            else:
                M_global = M_global + M_m  # disjoint blocks in your DG case, so this is cheap

        out_g = os.path.join(OUT_DIR, "mass_global_mat.npz")
        save_npz(out_g, M_global.tocsc())
        print("Saved:", out_g, " nnz=", M_global.nnz)

    else:
        print("SciPy not available (or SAVE_SPARSE=False).")
        print("You still have compact arrays (cell2pt, areas, materials, scale, template) to rebuild mass matrices.")

    # Print the reference/element matrix facts you asked about
    print("\nReference/element mass matrix check:")
    print("For any triangle of area A: M = (A/12) * [[2,1,1],[1,2,1],[1,1,2]]")
    print("If A_ref = 1/2, then M_ref = (1/24) * [[2,1,1],[1,2,1],[1,1,2]]")
    print("If A_ref = 1,   then M_ref = (1/12) * [[2,1,1],[1,2,1],[1,1,2]]")


# =====================================================================
# Plotting (merged from plotMassMatrix.py)
# NOTE: this is appended and does not alter the compute/save logic above.
# =====================================================================

def _natural_key(s):
    import re
    _num_re = re.compile(r"(\d+)")
    return [int(tok) if tok.isdigit() else tok.lower() for tok in _num_re.split(s)]


def _extract_mat_id(path):
    import re
    base = os.path.basename(path)
    m = re.search(r"mass_mat_(\d+)\.npz$", base)
    if m:
        return int(m.group(1))
    return None


def plot_saved_mass_matrices(
    mass_dir=OUT_DIR,
    pattern=PLOT_PATTERN,
    save_dir=None,
    markersize=PLOT_MARKERSIZE,
    dpi=PLOT_DPI,
    show=PLOT_SHOW,
):
    """
    Loads saved per-material matrices mass_mat_<matid>.npz and writes spy plots.
    """
    try:
        import glob
        import matplotlib.pyplot as plt
    except Exception as e:
        print("Plotting skipped: could not import matplotlib/glob. Error:", e)
        return

    try:
        from scipy.sparse import load_npz
    except Exception as e:
        print("Plotting skipped: could not import scipy.sparse.load_npz. Error:", e)
        return

    if save_dir is None:
        save_dir = os.path.join(mass_dir, PLOT_SAVE_DIRNAME)
    os.makedirs(save_dir, exist_ok=True)

    files = glob.glob(os.path.join(mass_dir, pattern))
    files.sort(key=_natural_key)

    if not files:
        print("Plotting skipped: no files matched:", os.path.join(mass_dir, pattern))
        print("Did you set SAVE_SPARSE=True and have SciPy installed?")
        return

    for fp in files:
        mat_id = _extract_mat_id(fp)
        if mat_id is None:
            print("Skipping (could not parse mat id):", fp)
            continue

        M = load_npz(fp)

        plt.figure()
        plt.spy(M, markersize=markersize)
        plt.title("Mass matrix sparsity (material {})\nshape={}, nnz={}".format(mat_id, M.shape, M.nnz))
        plt.xlabel("column")
        plt.ylabel("row")
        plt.tight_layout()

        out_png = os.path.join(save_dir, "mass_mat_{}_spy.png".format(mat_id))
        plt.savefig(out_png, dpi=dpi)
        print("Saved:", out_png)

        if show:
            plt.show()
        else:
            plt.close()

    print("Done. Plots in:", save_dir)


if __name__ == "__main__":
    main()
    if PLOT_AFTER_COMPUTE:
        plot_saved_mass_matrices()
