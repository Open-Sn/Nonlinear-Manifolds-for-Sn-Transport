# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 23:00:40 2026

@author: jean.ragusa
"""

# -*- coding: utf-8 -*-
"""
Load full-order spatial operators and project them with U to get reduced operators.

Inputs:
  - direction weights
  - full_order_out/ops_meta.npz (contains paths + fq + N)
  - ops_Mt.npz, ops_M1.npz, ops_Mv.npz, ops_Ms.npz, ops_Mf.npz (sparse CSC)
  - U memmapped .npy: shape (N*Ndir, K)

Outputs:
  - quad_forms_out/reduced_ops.npz containing:
        Gt, G1, Gv, Gs, Gf, f_red, plus metadata
"""


# ----------------------------
# Settings (edit these)
# ----------------------------
import os
import numpy as np
from scipy.sparse import load_npz
FULL_META_PATH = os.path.join("..", "run", "preparation", "full_order_out", "ops_meta.npz")
U_PATH = os.path.join("..", "run", "preparation", "svd_out", "U_modes_K120.npy")

OUT_DIR = os.path.join("..", "run", "preparation", "quad_forms_out")
OUT_NPZ = "reduced_ops.npz"

FORCE_FLOAT64 = True

W_PATH = "w.txt"

# ----------------------------
# Helpers
# ----------------------------


def read_weights(path):
    w = np.loadtxt(path, dtype=np.float64)
    w = np.atleast_1d(w).reshape(-1)
    return w


def ensure_dir(d):
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


"""
Block-diagonal action:
       BigM = I_{Ndir} ⊗ M
   Here I_{Ndir} has rank Ndir. The reduced form is
       U^T BigM U = Σ_d (U_d^T M U_d),
   so each direction block contributes independently, requiring Ndir SpMM calls M·U_d.
"""


def compute_Gt_blockdiag(U_mmap, Mt, N, Ndir, K):
    Gt = np.zeros((K, K), dtype=np.float64)

    for d in range(Ndir):
        i0 = d * N
        i1 = (d + 1) * N
        Ud = U_mmap[i0:i1, :]

        Ud64 = np.asarray(Ud, dtype=np.float64) if FORCE_FLOAT64 else Ud
        tmp = Mt.dot(Ud64)
        Gt += Ud64.T @ tmp

        # print('U64.shape=', Ud64.shape)
        # print('Mt.shape =', Mt.shape)
        # print('Gt.shape=', Gt.shape)

        if (d + 1) % 10 == 0 or (d + 1) == Ndir:
            print("Gt-like: processed {}/{} directions".format(d + 1, Ndir))

    return Gt


"""
For operators with identical block rows in angle space:

Let U be stacked by direction blocks U_d (shape N×K), so U = [U_0; U_1; ...; U_{Ndir-1}].
Given a spatial matrix M (N×N) and direction weights w (length Ndir), define the expanded
operator BigM (size (N·Ndir)×(N·Ndir)) by its blocks:

    BigM_{ij} = w[j] * M

i.e., every block row equals [w0*M, w1*M, ..., w_{Ndir-1}*M].

Then the reduced quadratic form can be computed without forming BigM:

    A = sum_d U_d
    B = sum_d w[d] * U_d
    U^T BigM U = A^T M B

This is the motivation for compute_A_B() and compute_G_identical_rows().
"""


def compute_A_B(U_mmap, w, N, Ndir, K):
    A = np.zeros((N, K), dtype=np.float64)
    B = np.zeros((N, K), dtype=np.float64)

    for d in range(Ndir):
        i0 = d * N
        i1 = (d + 1) * N
        Ud = U_mmap[i0:i1, :]

        Ud64 = np.asarray(Ud, dtype=np.float64) if FORCE_FLOAT64 else Ud
        A += Ud64
        B += float(w[d]) * Ud64

        if (d + 1) % 10 == 0 or (d + 1) == Ndir:
            print("A/B: processed {}/{} directions".format(d + 1, Ndir))

    return A, B


"""
For BigM = W ⊗ M with W_{ij} = w[j] (identical rows), the reduced form is:
   G = A^T M B

(1 w^T) has rank 1. This low-rank angular structure allows factorization:
   U^T BigM U = (Σ_d U_d)^T M (Σ_d w[d] U_d) = A^T M B,
so we only need one SpMM call M·B after forming the two aggregates A and B.
"""


def compute_G_identical_rows(A, B, M):
    tmp = M.dot(B)
    G = A.T @ tmp
    return G


def compute_forcing_reduced(U_mmap, fq, N, Ndir, K):
    f_red = np.zeros(K, dtype=np.float64)
    for d in range(Ndir):
        i0 = d * N
        i1 = (d + 1) * N
        Ud = U_mmap[i0:i1, :]

        Ud64 = np.asarray(Ud, dtype=np.float64) if FORCE_FLOAT64 else Ud
        f_red += Ud64.T @ fq

        if (d + 1) % 10 == 0 or (d + 1) == Ndir:
            print("forcing: processed {}/{} directions".format(d + 1, Ndir))

    return f_red


def main():
    ensure_dir(OUT_DIR)

    # Read weights (only for metadata; reduced script will need it)
    w = read_weights(W_PATH)
    Ndir = int(w.size)
    print("Ndir =", Ndir)

    # Load meta
    with np.load(FULL_META_PATH, allow_pickle=True) as z:
        N = int(z["N"])
        fq = z["fq"].astype(np.float64)

        Mt_path = str(z["Mt_path"])
        M1_path = str(z["M1_path"])
        Mv_path = str(z["Mv_path"])
        Ms_path = str(z["Ms_path"])
        Mf_path = str(z["Mf_path"])

        materials = z["materials"]
        sigt = z["sigt"]
        ivel = z["ivel"]
        sigs = z["sigs"]
        sigf = z["sigf"]
        qext = z["qext"]

    print("Loaded meta:", FULL_META_PATH)
    print("N =", N, " Ndir =", Ndir)

    # Load U
    U = np.load(U_PATH, mmap_mode="r")
    if U.ndim != 2:
        raise RuntimeError("U must be 2D. Got shape {}".format(U.shape))
    nrows, K = U.shape
    if nrows != N * Ndir:
        raise RuntimeError("U has {} rows, expected N*Ndir = {}".format(nrows, N * Ndir))
    print("Loaded U:", U_PATH, " shape=", U.shape)

    # Load operators
    Mt = load_npz(Mt_path).tocsc()
    M1 = load_npz(M1_path).tocsc()
    Mv = load_npz(Mv_path).tocsc()
    Ms = load_npz(Ms_path).tocsc()
    Mf = load_npz(Mf_path).tocsc()

    # Block-diagonal type reductions
    print("\nComputing Gt, G1, Gv ...")
    Gt = compute_Gt_blockdiag(U, Mt, N, Ndir, K)
    G1 = compute_Gt_blockdiag(U, M1, N, Ndir, K)
    Gv = compute_Gt_blockdiag(U, Mv, N, Ndir, K)

    # Identical-row weighted reductions
    print("\nComputing A,B ...")
    A, B = compute_A_B(U, w, N, Ndir, K)

    print("\nComputing Gs, Gf ...")
    Gs = compute_G_identical_rows(A, B, Ms)
    Gf = compute_G_identical_rows(A, B, Mf)

    # Forcing reduction
    print("\nComputing f_red ...")
    f_red = compute_forcing_reduced(U, fq, N, Ndir, K)

    # Save reduced results
    out_path = os.path.join(OUT_DIR, OUT_NPZ)
    np.savez_compressed(
        out_path,
        Gt=Gt, G1=G1, Gv=Gv, Gs=Gs, Gf=Gf,
        f_red=f_red.astype(np.float64),
        N=np.int64(N), Ndir=np.int64(Ndir), K=np.int64(K),
        weights=w.astype(np.float64),
        materials=materials,
        sigt=sigt, ivel=ivel, sigs=sigs, sigf=sigf, qext=qext,
        U_PATH=U_PATH,
        FULL_META_PATH=FULL_META_PATH,
        Mt_path=Mt_path, M1_path=M1_path, Mv_path=Mv_path, Ms_path=Ms_path, Mf_path=Mf_path
    )
    print("Saved reduced results ->", out_path)


if __name__ == "__main__":
    main()
