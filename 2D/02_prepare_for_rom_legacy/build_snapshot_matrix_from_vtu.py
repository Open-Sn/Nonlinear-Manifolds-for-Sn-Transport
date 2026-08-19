# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 21:37:09 2026

@author: jean.ragusa
"""

"""
Build a data matrix from many .vtu files, with OPTIONAL centering subtraction.

For each .vtu file:
  - find ALL PointData arrays whose name starts with FIELD_PREFIX (e.g. "psi_g000_a")
  - extract them in a consistent (natural-sorted) order
  - flatten each array and concatenate vertically into ONE column vector

Optional centering:
  - If CENTER_FILE is provided, read it once, build the same stacked column vector C,
    then subtract it from every file column:  X[:,j] = col_j - C.
  - If CENTER_FILE is None, behavior is identical to the previous script (no subtraction).

Then:
  - stack columns for all files into a 2D ndarray X with shape (n_dofs_per_file, n_files)
  - save compressed output (default: .npz via np.savez_compressed)

Requires VTK Python:
  pip install vtk
"""


# --- VTK imports (support both old-style vtk.* and newer vtkmodules.*) ---
try:
    import os
    import re
    import gc
    import glob
    import json
    import gzip
    import shutil
    import tempfile
    import numpy as np
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
    from vtkmodules.util.numpy_support import vtk_to_numpy
except ImportError:
    from vtk import vtkXMLUnstructuredGridReader  # type: ignore
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore


# ----------------------------
# Configuration (edit these)
# ----------------------------
INPUT_GLOB = "../run/opensn/transient/aflux_3newh_*_0.vtu"     # e.g., "/path/to/vtu/aflux_1proc_*_0.vtu"
FIELD_PREFIX = "psi_g000_a"            # matches psi_g000_a000_gs00 ... etc.
ASSOCIATION = "point"                 # "point" (as you said) or "cell"
DTYPE = np.float32                    # float32 saves RAM/disk; use np.float64 if needed
DTYPE = np.float64                    # float32 saves RAM/disk; use np.float64 if needed
USE_MEMMAP = True                     # recommended for 100s of files
OUTPUT_PATH = "../run/preparation/snapshots/psi_matrix_centered_fp64.npz"        # ".npz" (recommended) or ".npy" or ".npy.gz"

# Optional centering (subtract this stacked column from all columns):
# CENTER_FILE = None                    # e.g., "centering_solution.vtu" or None
CENTER_FILE = "../run/opensn/aflux_3newss_1000_0.vtu"

# Robustness checks:
REQUIRE_SAME_FIELDS = True            # recommended
REQUIRE_SAME_LENGTH = True            # recommended


# ----------------------------
# Helpers
# ----------------------------
_num_re = re.compile(r"(\d+)")


def natural_key(s):
    """Key for natural sorting (so a2 < a10)."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in _num_re.split(s)]


def read_vtu(path):
    r = vtkXMLUnstructuredGridReader()
    r.SetFileName(path)
    r.Update()
    return r.GetOutput()


def get_data_object(grid, association):
    association = association.lower().strip()
    if association == "point":
        return grid.GetPointData()
    if association == "cell":
        return grid.GetCellData()
    raise ValueError("Unknown association='{}' (use 'point' or 'cell')".format(association))


def list_matching_array_names(grid, prefix, association):
    data = get_data_object(grid, association)
    n = data.GetNumberOfArrays()
    names = []
    for i in range(n):
        name = data.GetArrayName(i)
        if name and name.startswith(prefix):
            names.append(name)
    names.sort(key=natural_key)
    return names


def extract_column_from_file(path, association, prefix, expected_names=None, dtype=np.float32):
    """
    Returns:
      col: 1D numpy array (concatenated fields)
      names: matched array names (sorted)
    """
    grid = read_vtu(path)
    data = get_data_object(grid, association)

    # Discover matching arrays if not provided
    if expected_names is None:
        names = list_matching_array_names(grid, prefix, association)
        if not names:
            raise RuntimeError(
                "No arrays matched prefix '{}' in {} data for file: {}".format(
                    prefix, association, path)
            )
    else:
        names = list(expected_names)

    # Concatenate flattened arrays
    parts = []
    for nm in names:
        arr = data.GetArray(nm)
        if arr is None:
            raise RuntimeError("Missing array '{}' in file: {}".format(nm, path))
        a = vtk_to_numpy(arr)          # shape: (npts,) or (npts, ncomp)
        a = np.asarray(a).reshape(-1)  # flatten
        parts.append(a.astype(dtype, copy=False))

    col = np.concatenate(parts, axis=0)
    return col, names


def save_npy_gz(array, out_path, chunk_mb=64):
    """
    Save a gzipped .npy (streaming via a temporary .npy).
    Output is a valid .npy file compressed with gzip: out_path should end with .npy.gz
    """
    assert out_path.endswith(".npy.gz")
    with tempfile.TemporaryDirectory() as td:
        tmp_npy = os.path.join(td, "tmp.npy")
        np.save(tmp_npy, array)
        with open(tmp_npy, "rb") as f_in, gzip.open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=chunk_mb * 1024 * 1024)


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    files = glob.glob(INPUT_GLOB)
    files.sort(key=natural_key)
    if not files:
        raise RuntimeError("No files matched INPUT_GLOB='{}'".format(INPUT_GLOB))

    # Discover field list and column length from the first input file
    col0, names0 = extract_column_from_file(
        files[0], ASSOCIATION, FIELD_PREFIX, expected_names=None, dtype=DTYPE)
    nrows = col0.size
    ncols = len(files)
    if ncols != 1001:
        raise RuntimeError("Expected 1001 snapshots, found {}".format(ncols))

    print("Matched {} field(s) in first file.".format(len(names0)))
    print("Column length per file: {}".format(nrows))
    print("Number of files: {}".format(ncols))
    print("Matrix shape will be: ({}, {})".format(nrows, ncols))

    # Optional centering vector
    center_col = None
    if CENTER_FILE is not None:
        center_col, center_names = extract_column_from_file(
            CENTER_FILE,
            ASSOCIATION,
            FIELD_PREFIX,
            expected_names=(names0 if REQUIRE_SAME_FIELDS else None),
            dtype=DTYPE
        )

        if REQUIRE_SAME_FIELDS and center_names != names0:
            raise RuntimeError(
                "Center file field-name mismatch.\nExpected: {}\nGot:      {}\nCenter file: {}".format(
                    names0, center_names, CENTER_FILE
                )
            )

        if REQUIRE_SAME_LENGTH and center_col.size != nrows:
            raise RuntimeError(
                "Center file column-length mismatch. Expected {}, got {}. Center file: {}".format(
                    nrows, center_col.size, CENTER_FILE
                )
            )

        print("Centering enabled using file: {}".format(CENTER_FILE))

    # Allocate matrix
    mm_path = None
    if USE_MEMMAP:
        mm_fd, mm_path = tempfile.mkstemp(prefix="vtu_stack_", suffix=".dat")
        os.close(mm_fd)
        X = np.memmap(mm_path, dtype=DTYPE, mode="w+", shape=(nrows, ncols))
    else:
        X = np.empty((nrows, ncols), dtype=DTYPE)

    # Fill columns
    for j, fp in enumerate(files):
        col, names = extract_column_from_file(
            fp,
            ASSOCIATION,
            FIELD_PREFIX,
            expected_names=(names0 if REQUIRE_SAME_FIELDS else None),
            dtype=DTYPE
        )

        if REQUIRE_SAME_FIELDS and names != names0:
            raise RuntimeError(
                "Field-name mismatch in file {}\nExpected: {}\nGot:      {}".format(
                    fp, names0, names)
            )

        if REQUIRE_SAME_LENGTH and col.size != nrows:
            raise RuntimeError(
                "Column-length mismatch in file {}\nExpected length {}, got {}".format(
                    fp, nrows, col.size)
            )

        if center_col is not None:
            col = col - center_col

        X[:, j] = col

        if (j + 1) % 10 == 0 or (j + 1) == ncols:
            print("Processed {}/{} files".format(j + 1, ncols))

    # Ensure memmap is written
    if USE_MEMMAP:
        X.flush()

    # Save output (recommended: .npz with metadata)
    out = OUTPUT_PATH
    if out.endswith(".npz"):
        np.savez_compressed(
            out,
            X=np.asarray(X),  # np.savez_compressed will read from memmap as needed
            files=np.array(files, dtype=object),
            field_names=np.array(names0, dtype=object),
            association=ASSOCIATION,
            field_prefix=FIELD_PREFIX,
            center_file=(CENTER_FILE if CENTER_FILE is not None else ""),
            dtype=str(np.dtype(DTYPE)),
            shape=np.array([nrows, ncols], dtype=np.int64),
        )
        print("Saved compressed matrix to: {}".format(out))

    elif out.endswith(".npy"):
        np.save(out, np.asarray(X))
        print("Saved matrix to: {}".format(out))

    elif out.endswith(".npy.gz"):
        save_npy_gz(np.asarray(X), out)
        meta_path = out + ".json"
        meta = {
            "files": files,
            "field_names": names0,
            "association": ASSOCIATION,
            "field_prefix": FIELD_PREFIX,
            "center_file": (CENTER_FILE if CENTER_FILE is not None else ""),
            "dtype": str(np.dtype(DTYPE)),
            "shape": [int(nrows), int(ncols)],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print("Saved gzipped .npy to: {}".format(out))
        print("Saved metadata to: {}".format(meta_path))

    else:
        raise ValueError("OUTPUT_PATH must end with .npz, .npy, or .npy.gz")

    # # Cleanup memmap backing file
    # if mm_path is not None and os.path.exists(mm_path):
    #     os.remove(mm_path)
    # Cleanup memmap backing file (Windows requires closing the mmap first)
    if mm_path is not None:
        try:
            # Make sure data is written
            try:
                X.flush()
            except Exception:
                pass

            # Close the underlying mmap handle if present
            try:
                if hasattr(X, "_mmap") and X._mmap is not None:
                    X._mmap.close()
            except Exception:
                pass

            # Remove references so Windows releases the file
            del X
            gc.collect()

            # Now try deleting
            try:
                os.remove(mm_path)
            except PermissionError:
                print("Note: could not delete memmap temp file on Windows. Delete manually if desired:")
                print(mm_path)

        except Exception as e:
            print("Note: memmap cleanup issue:", e)


if __name__ == "__main__":
    main()
