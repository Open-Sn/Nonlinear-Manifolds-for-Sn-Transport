# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 17:07:11 2023

@author: jean.ragusa
"""
import numpy as np


class MESH:
    def __init__(self, width, n_ref, verbose=False):

        if len(width) != len(n_ref):
            raise Exception("MESH::width and nref must be of the same length")
        # number of materials
        nzones = len(width)
        # number of cells
        ncells = np.sum(n_ref)

        # cell width
        cell2mat = np.zeros(ncells, dtype=int)

        # mapping from cell ID to material ID
        ibeg = 0
        for m in range(nzones):
            # width in that mateiral zone
            # dx[m] = width[m] / n_ref[m]
            # create cellID to matID mapping
            iend = ibeg + n_ref[m]
            if verbose:
                print("MESH::range for current material:", ibeg, iend)
            cell2mat[ibeg:iend] = m
            ibeg = iend
        # cell interfaces
        x0 = 0.0
        for m in range(nzones):
            x1 = np.sum(width[: m + 1])
            aux = np.linspace(x0, x1, n_ref[m] + 1)
            if m == 0:
                x = np.copy(aux)
            else:
                x = np.append(x, aux[1:])
            x0 = x1
        # cell mid-points
        xm = (x[1:] + x[:-1]) / 2.0

        # save in object
        self.n_zones = nzones
        self.n_cells = ncells
        self.dx = np.diff(x)
        self.x = np.copy(x)
        self.cell2mat = np.copy(cell2mat)


if __name__ == "__main__":
    print("Running MESH as the main code:")

    width = np.array([2.0, 1.0, 2.0, 1.0, 2.0])
    n_ref = np.array([4, 5, 3, 3, 2], dtype=int)

    myMESH = MESH(width, n_ref)
    print("ncells = ", myMESH.ncells)
    print("nzones = ", myMESH.nzones)
    print("xm = ", myMESH.xm)
