# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 22:07:02 2026

@author: jean.ragusa
"""
import os
import numpy as np
from scipy.sparse import load_npz

"""
Compute singular values / vectors of a snapshot matrix D stored in a compressed .npz.

Method:
  M = D^T D  (small: ncols x ncols)
  M v_i = lambda_i v_i
  s_i = sqrt(lambda_i)
  V = [v_i] are right singular vectors
  U_i = (D v_i) / s_i  (left singular vectors)

Notes:
- Loading .npz is not memory-mappable; it will decompress in memory.
  If D is huge, consider converting once to an uncompressed .npy for future runs.
- U can be enormous (nrows x ncols). Use K to compute only first modes.
"""

# ----------------------------
# Configuration (edit these)
# ----------------------------

FULL_META_PATH = os.path.join("..", "run", "preparation", "full_order_out", "ops_meta.npz")

NPZ_PATH = os.path.join("..", "run", "preparation", "snapshots", "psi_matrix_centered_fp64.npz")   # your saved snapshot matrix
MATRIX_KEY = "X"                # in our earlier writer: X=...
OUT_DIR = os.path.join("..", "run", "preparation", "svd_out")
# Save singular values separately + plot
SAVE_SVALS_NPY = True
SAVE_SVALS_CSV = True
SAVE_SVALS_PLOT = True

# Filenames (will include the effective K used)
SVALS_NPY_NAME = None   # if None, will auto-name based on K
SVALS_CSV_NAME = None
SVALS_PLOT_NAME = None   # e.g. .png

# If K is None, compute all modes (can be huge for U).
# Strongly recommended: set K to something like 20, 50, 100.
K = 120

# Row blocking for building C (if M1 not given) and for computing U (tune to your RAM)
BLOCK_ROWS = 200_000

# Numerical threshold for "nonzero" singular values
S_TOL = 1e-14

# Save U as a separate .npy memmap (recommended)
SAVE_U_MEMMAP = True
# written under OUT_DIR
# U_NPY_NAME = f"U_modes_K{K}.npy" if K is not None else "U_modes_Kall.npy"
U_NPY_NAME = None

# Save s and V to an .npz
RESULTS_NPZ_NAME = "svd_results_fp32.npz"


# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def load_D_from_npz(npz_path, key="X"):
    z = np.load(npz_path, allow_pickle=True)
    if key not in z:
        raise KeyError("Key '{}' not found in {}. Available keys: {}".format(
            key, npz_path, list(z.keys())))
    D = z[key]
    meta = {}
    for k in ("files", "field_names", "association", "field_prefix", "center_file", "dtype", "shape"):
        if k in z:
            meta[k] = z[k]
    return D, meta


def compute_gram_matrix(D, M1=None, block_rows=200_000):
    """
    Compute C = D^T D in float64 accumulation using row blocks.
    D is (nrows, ncols). Returns M shape (ncols, ncols).
    """
    nrows, ncols = D.shape
    C = np.zeros((ncols, ncols), dtype=np.float64)

    # -------------------------
    # Unweighted: Euclidean
    # -------------------------
    if M1 is None:
        for i0 in range(0, nrows, block_rows):
            i1 = min(nrows, i0 + block_rows)
            Db = np.asarray(D[i0:i1, :], dtype=np.float64)  # block cast only
            C += Db.T @ Db
            # progress
            if (i0 // block_rows) % 5 == 0 or i1 == nrows:
                print("Gram: processed rows [{}, {}) / {}".format(i0, i1, nrows))
    else:
        # -------------------------
        # Weighted: BigM = I ⊗ M1
        # -------------------------
        N = int(M1.shape[0])
        if M1.shape[1] != N:
            raise RuntimeError("M1 must be square. Got shape {}".format(M1.shape))

        if nrows % N != 0:
            raise RuntimeError("Expected D.shape[0] = N*Ndir. Got nrows={}, N={}".format(nrows, N))

        Ndir = nrows // N

        # CSR is typically faster for sparse @ dense
        try:
            M1_op = M1.tocsr()
        except Exception:
            M1_op = M1

        for d in range(Ndir):
            i0 = d * N
            i1 = (d + 1) * N

            Dd = np.asarray(D[i0:i1, :], dtype=np.float64)  # (N, nsnaps)
            Y = M1_op.dot(Dd)                               # (N, nsnaps)
            C += Dd.T @ Y                                   # (nsnaps, nsnaps)

            if (d + 1) % 10 == 0 or (d + 1) == Ndir:
                print("Weighted Gram: processed {}/{} directions".format(d + 1, Ndir))

    # enforce symmetry (roundoff)
    C = 0.5 * (C + C.T)
    return C


def eig_sorted_desc(M):
    """
    For symmetric M: returns eigenvalues (descending) and eigenvectors (columns) aligned.
    """
    evals, evecs = np.linalg.eigh(M)  # ascending
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    return evals, evecs


def compute_U_modes(D, V, s, block_rows=200_000, out_npy_path=None):
    """
    Compute U = D @ V / s for provided V (ncols x K) and s (K,).
    Uses row-blocking. If out_npy_path is given, writes U to a memmapped .npy.
    Returns U (either ndarray or memmap).
    """
    nrows, ncols = D.shape
    K = V.shape[1]

    if out_npy_path is not None:
        # Create a .npy on disk without allocating all in RAM
        U = np.lib.format.open_memmap(out_npy_path, mode="w+", dtype=np.float32, shape=(nrows, K))
    else:
        U = np.empty((nrows, K), dtype=np.float32)

    inv_s = 1.0 / s

    for i0 in range(0, nrows, block_rows):
        i1 = min(nrows, i0 + block_rows)
        Db = np.asarray(D[i0:i1, :], dtype=np.float64)   # block in float64 for stability
        Ub = Db @ V                                      # (block_rows x K)
        Ub *= inv_s                                      # divide each column by s
        U[i0:i1, :] = Ub.astype(np.float32, copy=False)

        if (i0 // block_rows) % 5 == 0 or i1 == nrows:
            print("U: processed rows [{}, {}) / {}".format(i0, i1, nrows))

    # flush if memmap
    try:
        U.flush()
    except Exception:
        pass

    return U


def save_and_plot_singular_values(s, out_dir, k_eff, plot_logy=True):
    # Auto-names
    npy_name = "singular_values_K{}.npy".format(k_eff)
    csv_name = "singular_values_K{}.csv".format(k_eff)
    plot_name = "singular_values_K{}.png".format(k_eff)

    npy_path = os.path.join(out_dir, npy_name)
    csv_path = os.path.join(out_dir, csv_name)
    plot_path = os.path.join(out_dir, plot_name)

    # Save .npy
    np.save(npy_path, s.astype(np.float64, copy=False))

    # Save .csv (index,value)
    idx = np.arange(1, s.size + 1, dtype=np.int64)
    arr = np.column_stack([idx, s.astype(np.float64, copy=False)])
    np.savetxt(csv_path, arr, delimiter=",", header="index,singular_value", comments="")

    # Plot and save
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(idx, s/s[0])
    plt.xlabel("Mode index")
    plt.ylabel("Singular value")
    if plot_logy:
        plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=400)
    plt.close()

    print("Saved singular values:", npy_path)
    print("Saved singular values CSV:", csv_path)
    print("Saved singular values plot:", plot_path)

# ----------------------------
# Main
# ----------------------------


def main():
    ensure_dir(OUT_DIR)

    # Load M1
    with np.load(FULL_META_PATH, allow_pickle=True) as z:
        M1_path = str(z["M1_path"])
        M1 = load_npz(M1_path).tocsc()
        # N = M1.shape[0]

    D, meta = load_D_from_npz(NPZ_PATH, MATRIX_KEY)
    if D.ndim != 2:
        raise ValueError("Expected a 2D matrix D, got shape {}".format(D.shape))

    nrows, ncols = D.shape
    print("Loaded D from {} key='{}' with shape {}".format(NPZ_PATH, MATRIX_KEY, D.shape))
    print("dtype:", D.dtype)

    # 1) Build Gram matrix
    C = compute_gram_matrix(D, M1=M1)
    print("Built M with shape", C.shape)

    # 2) Eigendecomposition of M
    evals, Vfull = eig_sorted_desc(C)

    # 3) Singular values
    # Clip negatives due to roundoff before sqrt
    evals_clipped = np.clip(evals, 0.0, None)
    sfull = np.sqrt(evals_clipped)

    # Plot the complete singular-value spectrum
    import matplotlib.pyplot as plt
    mode_index = np.arange(1, sfull.size + 1)
    plt.figure()
    plt.semilogy(mode_index, sfull)
    plt.axvline(K, color="k", linestyle="--", label="K = {}".format(K))
    plt.xlabel("POD/SVD mode index")
    plt.ylabel(r"Singular value $\sigma_i$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "singular_values.pdf"))
    plt.close()

    # Determine how many modes are effectively nonzero
    keep = np.where(sfull > S_TOL)[0]
    if keep.size == 0:
        raise RuntimeError("All singular values are ~0 (threshold {}).".format(S_TOL))

    r = keep.size
    if K is None:
        K_eff = r
    else:
        K_eff = int(min(K, r))

    print("Nonzero modes (s > {}): {}".format(S_TOL, r))
    print("Computing / saving first K = {} modes".format(K_eff))

    V = Vfull[:, :K_eff].astype(np.float64, copy=False)  # right singular vectors
    s = sfull[:K_eff].astype(np.float64, copy=False)
    if SAVE_SVALS_NPY or SAVE_SVALS_CSV or SAVE_SVALS_PLOT:
        save_and_plot_singular_values(s, OUT_DIR, K_eff, plot_logy=True)

    # 4) Compute U = D V / s
    U_path = None
    U = None

    if SAVE_U_MEMMAP:
        U_path = os.path.join(OUT_DIR, f"U_modes_K{K_eff}.npy")
        print("Writing U to memmap .npy:", U_path)
        U = compute_U_modes(D, V, s, block_rows=BLOCK_ROWS,
                            out_npy_path=U_path)  # U is a live memmap handle
    else:
        U = compute_U_modes(D, V, s, block_rows=BLOCK_ROWS, out_npy_path=None)   # U is in RAM

    # # 5) Compute expansion coefficients C = U^T D and save them
    # #    U is (nrows x K_eff), D is (nrows x ncols) => C is (K_eff x ncols)
    # Coef_path = os.path.join(OUT_DIR, "coeffs_C_K{}.npy".format(K_eff))
    # print("\nComputing coefficients C = U^T D -> shape ({}, {})".format(K_eff, ncols))
    # print("Writing C to:", Coef_path)

    # Coef = np.lib.format.open_memmap(Coef_path, mode="w+", dtype=np.float64, shape=(K_eff, ncols))
    # Coef[:, :] = 0.0

    # for i0 in range(0, nrows, BLOCK_ROWS):
    #     i1 = min(nrows, i0 + BLOCK_ROWS)
    #     Db = np.asarray(D[i0:i1, :], dtype=np.float64)        # (br, ncols)
    #     Ub = np.asarray(U[i0:i1, :], dtype=np.float64)        # (br, K_eff) from RAM or memmap
    #     Coef[:, :] += Ub.T @ Db                                  # accumulate (K_eff, ncols)

    #     if (i0 // BLOCK_ROWS) % 5 == 0 or i1 == nrows:
    #         print("C: processed rows [{}, {}) / {}".format(i0, i1, nrows))

    # try:
    #     Coef.flush()
    # except Exception:
    #     pass

    # # Close C memmap handle (important on Windows)
    # try:
    #     if hasattr(Coef, "_mmap") and Coef._mmap is not None:
    #         Coef._mmap.close()
    # except Exception:
    #     pass
    # Coef = None

    # # Now that C is done, we can close U if it was memmapped
    # if SAVE_U_MEMMAP:
    #     try:
    #         U.flush()
    #     except Exception:
    #         pass
    #     try:
    #         if hasattr(U, "_mmap") and U._mmap is not None:
    #             U._mmap.close()
    #     except Exception:
    #         pass
    #     U = None
    # 5) Compute expansion coefficients:
    #    Unweighted:      C = U^T D
    #    M-weighted:      C = U^T (I ⊗ M1) D  = sum_d U_d^T (M1 D_d)
    #
    #    U is (nrows x K_eff), D is (nrows x ncols) => C is (K_eff x ncols)

    Coef_path = os.path.join(OUT_DIR, "coeffs_C_K{}.npy".format(K_eff))
    print("\nComputing coefficients -> shape ({}, {})".format(K_eff, ncols))
    print("Writing C to:", Coef_path)

    Coef = np.lib.format.open_memmap(Coef_path, mode="w+", dtype=np.float64, shape=(K_eff, ncols))
    Coef[:, :] = 0.0

    # If M1 is provided, we MUST block by direction: block size = N
    if M1 is not None:
        N = int(M1.shape[0])
        if M1.shape[1] != N:
            raise RuntimeError("M1 must be square. Got shape {}".format(M1.shape))
        if nrows % N != 0:
            raise RuntimeError("Expected nrows = N*Ndir. Got nrows={}, N={}".format(nrows, N))

        BLOCK = N
        try:
            M1_op = M1.tocsr()   # faster for sparse @ dense
        except Exception:
            M1_op = M1

        print("Using M-weighted coefficients: C = U^T (I ⊗ M1) D with block size N =", N)

    else:
        BLOCK = BLOCK_ROWS
        M1_op = None
        print("Using Euclidean coefficients: C = U^T D with block_rows =", BLOCK)

    for i0 in range(0, nrows, BLOCK):
        i1 = min(nrows, i0 + BLOCK)

        Db = np.asarray(D[i0:i1, :], dtype=np.float64)   # (BLOCK, ncols)
        Ub = np.asarray(U[i0:i1, :], dtype=np.float64)   # (BLOCK, K_eff)

        if M1_op is None:
            # Euclidean: C += U_b^T D_b
            Coef[:, :] += Ub.T @ Db
        else:
            # Weighted: C += U_b^T (M1 D_b)
            # Here BLOCK == N so each block is one direction slice
            Yb = M1_op.dot(Db)          # (N, ncols)
            Coef[:, :] += Ub.T @ Yb     # (K_eff, ncols)

        if (i0 // BLOCK) % 10 == 0 or i1 == nrows:
            print("C: processed rows [{}, {}) / {}".format(i0, i1, nrows))

    try:
        Coef.flush()
    except Exception:
        pass

    # Close C memmap handle (important on Windows)
    try:
        if hasattr(Coef, "_mmap") and Coef._mmap is not None:
            Coef._mmap.close()
    except Exception:
        pass
    Coef = None

    # Now that C is done, we can close U if it was memmapped
    if SAVE_U_MEMMAP:
        try:
            U.flush()
        except Exception:
            pass
        try:
            if hasattr(U, "_mmap") and U._mmap is not None:
                U._mmap.close()
        except Exception:
            pass
        U = None

    # 6) Save results (singular values + V + metadata + pointers to U and C)
    out_npz = os.path.join(OUT_DIR, RESULTS_NPZ_NAME)
    np.savez_compressed(
        out_npz,
        singular_values=s.astype(np.float64),
        eigenvalues=evals[:K_eff].astype(np.float64),
        V=V.astype(np.float64),
        U_path=(U_path if U_path is not None else ""),
        Coef_path=Coef_path,
        D_shape=np.array([nrows, ncols], dtype=np.int64),
        npz_source=NPZ_PATH,
        matrix_key=MATRIX_KEY,
        **{("meta_"+k): meta[k] for k in meta}
    )

    print("Saved:", out_npz)
    if U_path is not None:
        print("Saved U (memmap .npy):", U_path)
    print("Saved C (memmap .npy):", Coef_path)

    print("\nTop singular values:")
    print(s[:min(10, s.size)])


if __name__ == "__main__":
    main()
