#!/usr/bin/env python3
"""Check U.T (I_ndir kron M1) U against the identity.

This is the pre-ROM mass-orthonormality check extracted from the maintained
2026-08-02 validation workflow. It reads existing POD modes and the full-order
spatial mass matrix; it does not construct or project transport operators.
"""

import argparse
import json

import numpy as np
from scipy.sparse import load_npz


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis", default="svd_out/U_modes_K80.npy")
    parser.add_argument("--mass", default="full_order_out/ops_M1.npz")
    parser.add_argument("--ndir", type=int, default=32)
    parser.add_argument("--rank", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    basis = np.load(args.basis, mmap_mode="r", allow_pickle=False)
    mass = load_npz(args.mass).tocsr()
    if basis.ndim != 2:
        raise ValueError("basis must be a two-dimensional array")
    if basis.shape[0] != args.ndir * mass.shape[0]:
        raise ValueError(
            "basis rows must equal ndir times the spatial mass dimension"
        )

    rank = basis.shape[1] if args.rank is None else args.rank
    if rank < 1 or rank > basis.shape[1]:
        raise ValueError("rank must be between 1 and the stored basis rank")

    gram = np.zeros((rank, rank), dtype=np.float64)
    n = mass.shape[0]
    for direction in range(args.ndir):
        block = np.asarray(
            basis[direction * n : (direction + 1) * n, :rank],
            dtype=np.float64,
        )
        gram += block.T @ mass.dot(block)

    error = gram - np.eye(rank)
    off_diagonal = error.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    report = {
        "basis": args.basis,
        "mass": args.mass,
        "ndir": args.ndir,
        "rank": rank,
        "maximum_absolute_entry_error": float(np.max(np.abs(error))),
        "frobenius_error": float(np.linalg.norm(error)),
        "diagonal_minimum": float(np.diag(gram).min()),
        "diagonal_maximum": float(np.diag(gram).max()),
        "maximum_absolute_off_diagonal": float(
            np.max(np.abs(off_diagonal)) if off_diagonal.size else 0.0
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
