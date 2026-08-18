# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 23:00:08 2026

@author: jean.ragusa
"""

# -*- coding: utf-8 -*-
"""
Build full-order spatial operators from per-material DG mass matrices.

Outputs:
  - Sparse CSC matrices: Mt, M1, Mv, Ms, Mf, Mq saved as .npz
  - Spatial forcing vector fq = Mq @ 1 saved in metadata .npz
  - Metadata (N, Ndir, materials, weights, coefficient arrays, paths)

This script does NOT load U and does NOT compute reduced quantities.
"""


# ----------------------------
# Settings (edit these)
# ----------------------------
import os
import re
import glob
import numpy as np
import scipy.sparse as sp
from scipy.sparse import load_npz, save_npz
MASS_DIR = "../run/preparation/dg_mass_out"
MASS_PATTERN = "mass_mat_*.npz"

SIGT_PATH = "sigt.txt"
IVEL_PATH = "ivel.txt"
SIGS_PATH = "sigs.txt"
SIGF_PATH = "sigf.txt"
QEXT_PATH = "qext.txt"

OUT_DIR = "../run/preparation/full_order_out"
OPS_PREFIX = "ops"  # files will be ops_Mt.npz, ops_meta.npz, etc.


# ----------------------------
# Helpers
# ----------------------------
_num_re = re.compile(r"(\d+)")


def natural_key(s):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in _num_re.split(s)]


def ensure_dir(d):
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def parse_mat_id_from_filename(path):
    base = os.path.basename(path)
    m = re.search(r"mass_mat_(\d+)\.npz$", base)
    return int(m.group(1)) if m else None


def read_theta_map(path):
    arr = np.loadtxt(path, dtype=np.float64)
    arr = np.atleast_2d(arr)
    if arr.shape[1] < 2:
        raise RuntimeError("File '{}' must have at least 2 columns: mat_id value".format(path))
    d = {}
    for row in arr:
        d[int(row[0])] = float(row[1])
    return d


def load_mass_matrices_csc(mass_dir, pattern):
    files = glob.glob(os.path.join(mass_dir, pattern))
    files.sort(key=natural_key)
    if not files:
        raise RuntimeError("No mass matrices found: {}".format(os.path.join(mass_dir, pattern)))

    mats = {}
    N = None

    # loop over filepaths (fp)
    for fp in files:
        mid = parse_mat_id_from_filename(fp)
        if mid is None:
            continue

        M = load_npz(fp).tocsc()

        if N is None:
            N = M.shape[0]
        if M.shape != (N, N):
            raise RuntimeError("Matrix {} has shape {}, expected {}x{}".format(fp, M.shape, N, N))

        mats[mid] = M
        print("Loaded material {}: {} (format={}, nnz={})".format(
            mid, os.path.basename(fp), M.getformat(), M.nnz
        ))

    if not mats:
        raise RuntimeError("Found files but could not parse any material ids from filenames.")

    return mats, N


def build_weighted_mass_csc(mats_dict, theta_dict, name="theta"):
    mids = sorted(mats_dict.keys())
    missing = [m for m in mids if m not in theta_dict]
    if missing:
        raise RuntimeError("Missing {} values for material ids: {}".format(name, missing))

    N = next(iter(mats_dict.values())).shape[0]
    Msum = sp.csc_matrix((N, N), dtype=np.float64)

    for m in mids:
        coef = float(theta_dict[m])
        if coef == 0.0:
            continue
        Msum = Msum + coef * mats_dict[m]

    return Msum.tocsc()


def save_op(out_dir, prefix, name, M):
    path = os.path.join(out_dir, "{}_{}.npz".format(prefix, name))
    save_npz(path, M.tocsc())
    print("Saved {} -> {}".format(name, path))
    return path


def main():
    ensure_dir(OUT_DIR)

    # Load per-material mass matrices
    mats_dict, N = load_mass_matrices_csc(MASS_DIR, MASS_PATTERN)
    mids = sorted(mats_dict.keys())
    print("Materials found:", mids)
    print("N (spatial dofs) =", N)

    # Read theta maps
    sigt = read_theta_map(SIGT_PATH)
    ivel = read_theta_map(IVEL_PATH)
    sigs = read_theta_map(SIGS_PATH)
    sigf = read_theta_map(SIGF_PATH)
    qext = read_theta_map(QEXT_PATH)
    unity = {m: 1.0 for m in mids}

    # Build spatial matrices
    Mt = build_weighted_mass_csc(mats_dict, sigt, name="sigt")
    M1 = build_weighted_mass_csc(mats_dict, unity, name="unity")
    Mv = build_weighted_mass_csc(mats_dict, ivel, name="ivel")
    Ms = build_weighted_mass_csc(mats_dict, sigs, name="sigs")
    Mf = build_weighted_mass_csc(mats_dict, sigf, name="sigf")
    Mq = build_weighted_mass_csc(mats_dict, qext, name="qext")
    # print(Mt.shape)
    # print(Mf.shape)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.spy(M1)
    plt.show()
    plt.figure()
    plt.spy(Mt)
    plt.show()
    plt.figure()
    plt.spy(Ms)
    plt.show()

    # Spatial forcing vector
    ones = np.ones(N, dtype=np.float64)
    fq = Mq.dot(ones)

    # Save matrices
    p_Mt = save_op(OUT_DIR, OPS_PREFIX, "Mt", Mt)
    p_M1 = save_op(OUT_DIR, OPS_PREFIX, "M1", M1)
    p_Mv = save_op(OUT_DIR, OPS_PREFIX, "Mv", Mv)
    p_Ms = save_op(OUT_DIR, OPS_PREFIX, "Ms", Ms)
    p_Mf = save_op(OUT_DIR, OPS_PREFIX, "Mf", Mf)
    p_Mq = save_op(OUT_DIR, OPS_PREFIX, "Mq", Mq)
    print('\nSaving Mq but technically not needed. Use fq instead.\n')

    # Save metadata + fq + coefficient arrays
    meta_path = os.path.join(OUT_DIR, "{}_meta.npz".format(OPS_PREFIX))
    np.savez_compressed(
        meta_path,
        N=np.int64(N),
        materials=np.array(mids, dtype=np.int64),
        sigt=np.array([sigt[m] for m in mids], dtype=np.float64),
        ivel=np.array([ivel[m] for m in mids], dtype=np.float64),
        sigs=np.array([sigs[m] for m in mids], dtype=np.float64),
        sigf=np.array([sigf[m] for m in mids], dtype=np.float64),
        qext=np.array([qext[m] for m in mids], dtype=np.float64),
        fq=fq.astype(np.float64),
        MASS_DIR=MASS_DIR,
        MASS_PATTERN=MASS_PATTERN,
        SIGT_PATH=SIGT_PATH,
        IVEL_PATH=IVEL_PATH,
        SIGS_PATH=SIGS_PATH,
        SIGF_PATH=SIGF_PATH,
        QEXT_PATH=QEXT_PATH,
        Mt_path=p_Mt, M1_path=p_M1, Mv_path=p_Mv, Ms_path=p_Ms, Mf_path=p_Mf, Mq_path=p_Mq,
    )
    print("Saved meta ->", meta_path)


if __name__ == "__main__":
    main()
