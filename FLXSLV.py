# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 17:07:11 2023

@author: jean.ragusa
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import gmres, LinearOperator
import scipy.sparse


# %%
def callback(rk):
    """
    Callback function supplied to gmres to count iterations.
    Args :
     rk : r esidu al vec tor passed by gmres .
    """
    global gmres_iter
    gmres_iter += 1


# %%
class gmres_counter(object):  # counter class to get gmres info
    def __init__(self, disp=True):  # constructor
        self._disp = disp  # set display
        self.niter = 0  # set number of iterations to zero

    def __call__(self, rk=None):  # callback function
        self.niter += 1  # increment the iteration counter
        if self._disp:  # if display is set
            print("\titer %3i\trk = %s" % (self.niter, str(rk)))  # print the iterate info


# %%
class FLXSLV:
    # %%
    def __init__(self, AQ, MSH, XS, SRC, mass_lumping=False):
        # note that XS and SRC are not saved in FLXSLV as they are but their contents is
        # save at the end of the __init__() routine.
        self.AQ = AQ
        self.MSH = MSH

        self.mass_lumping = mass_lumping
        self.n_dofs = 2 * MSH.n_cells

        # check XS
        if not isinstance(XS, dict):
            raise Exception("FLXSLV::XS must be given as a dictionary")
        keylist = {"sigs", "sigt"}
        for key in keylist:
            # check array length
            if key in XS:
                if len(XS[key]) != MSH.n_zones:
                    raise Exception("FLXSLV::XS[{0}] does not have enough materials".format(key))
            else:
                raise Exception(
                    "FLXSLV::key {0} is missing from XS dictionary. \n\tFound {1}".format(
                        key, XS.keys()
                    )
                )
        # check positivity of XS and check that sigs is not > than sigt
        for m, (sigt, sigs) in enumerate(zip(XS["sigt"], XS["sigs"])):
            if sigs < 0:
                raise Exception("sigs={0} is <0 for material {1}".format(sigs, m))
            if sigt < 0:
                raise Exception("sigt={0} is <0 for material {1}".format(sigt, m))
            # if sigs > sigt:
            #     raise Exception(
            #         "sigs={0} is > than sigt {1} for material {2}".format(sigs, sigt, m)
            #     )
        # check SRC
        if not isinstance(SRC, dict):
            raise Exception("FLXSLV::SRC must be given as a dictionary")
        keylist = {"qext", "psi_bc"}
        for key in keylist:
            if not key in SRC:
                raise Exception(
                    "FLXSLV::key {0} is missing from SRC dictionary. \n\tFound {1}".format(
                        key, SRC.keys()
                    )
                )
        # check array length
        if len(SRC["qext"]) != MSH.n_zones:
            raise Exception("FLXSLV::SRC[qext] does not have enough materials")
        if len(SRC["psi_bc"]) != AQ.ndir:
            raise Exception("FLXSLV::SRC[psi_bc] does not have enough directions")
        self.psi_bc = np.copy(SRC["psi_bc"])

        # check positivity
        for m, qext in enumerate(SRC["qext"]):
            if qext < 0:
                raise Exception("qext={0} is <0 for material {1}".format(qext, m))
        for d, psibc in enumerate(SRC["psi_bc"]):
            if psibc < 0:
                raise Exception("psibc={0} is <0 for direction {1}".format(psibc, d))
        # prepare arrays
        self.sigt = np.zeros(MSH.n_cells)
        for i, matID in enumerate(MSH.cell2mat):
            self.sigt[i] = XS["sigt"][matID]
        # build local matrices
        self.m = np.array([[2, 1], [1, 2]], dtype=float) / 3.0
        # self.k = np.array([[1, -1], [-1, 1]], dtype=float) / 2.0
        self.g = np.array([[1, 1], [-1, -1]], dtype=float) / 2.0
        self.f = np.array([1, 1], dtype=float)

        # activate lumping if requested
        if self.mass_lumping:
            row_sums = np.sum(self.m, axis=1)  # Sum over rows
            self.m = np.diag(row_sums)  # Create a diagonal matrix
        self.compute_mass_matrices_by_material()
        self.compute_source_vectors_by_material()

        self.qext = self.assemble_global_qext_vector(SRC["qext"])
        self.M_scattering = self.assemble_global_mass_matrix(XS["sigs"])
        siga = XS["sigt"] - XS["sigs"]
        self.M_abso = self.assemble_global_mass_matrix(siga)

    # %%
    def assemble_global_grad_matrix(self, psi_bc):
        """
        Constructs a block diagonal matrix of size 2N x 2N where each 2x2 block
        is given by the Kronecker product of self.m and dx[iel].
        """

        # size of the global system
        n = self.AQ.ndir * self.n_dofs
        # global Gradient matrix
        G = scipy.sparse.lil_matrix((n, n))
        # right due to boundary conditions
        q_surf = np.zeros((n, self.AQ.ndir))

        n_cells = self.MSH.n_cells
        mat = np.zeros(((2, 2)))

        # loop over all directions
        for dir_ in range(self.AQ.ndir):
            offset = dir_ * self.n_dofs
            # select direction and sweep order
            mu = self.AQ.mu_q[dir_]
            if mu > 0:
                ibeg, iend, incr = 0, n_cells, +1
            else:
                ibeg, iend, incr = n_cells - 1, -1, -1

            # bc
            psi_in = psi_bc[dir_]

            # loop over cells
            for iel in range(ibeg, iend, incr):
                jac = self.MSH.dx[iel] / 2.0
                mat[:, :] = mu * self.g
                # LOCAL → GLOBAL indexing:
                i1 = offset + 2 * iel
                i2 = offset + 2 * iel + 1
                if mu > 0:
                    mat[1, 1] += mu
                    if iel > 0:
                        G[i1, i1 - 1] = -mu
                    else:
                        q_surf[i1, dir_] = psi_in * mu
                else:
                    mat[0, 0] -= mu
                    if iel < n_cells - 1:
                        G[i2, i2 + 1] = mu
                    else:
                        q_surf[i2, dir_] = -psi_in * mu
                G[i1 : i2 + 1, i1 : i2 + 1] = mat

        return G.tocsc(), q_surf

    # %%
    def compute_mass_matrix(self):
        """
        Constructs a block diagonal matrix of size 2N x 2N where each 2x2 block
        is given by the Kronecker product of self.m and dx[iel].
        """
        # Initialize the list of blocks
        blocks = []

        for iel in range(self.MSH.n_cells):
            # Compute the scaled local matrix for the current cell
            local_A = self.m * self.MSH.dx[iel] / 2.0
            # Append to the list of blocks
            blocks.append(local_A)
        # Create the block diagonal matrix
        A = scipy.sparse.block_diag(blocks, format="csc")

        return A

    # %%
    def compute_mass_matrices_by_material(self):
        """
        Constructs a list of mass matrices, one per unique material ID.
        Each matrix corresponds to a block diagonal matrix with non-zero entries
        only for the cells of the current material.
        """
        # Extract unique material IDs
        unique_material_ids = np.unique(self.MSH.cell2mat)

        # List to store mass matrices for each material
        self.material_mass_matrices = []

        # Outer loop over unique material IDs
        for material_id in unique_material_ids:
            # Initialize a list of blocks for the current material
            blocks = []

            for iel in range(self.MSH.n_cells):
                # Check if the current cell belongs to the material
                if self.MSH.cell2mat[iel] == material_id:
                    # Compute the scaled local matrix for the current cell
                    local_A = self.m * self.MSH.dx[iel] / 2.0
                else:
                    # Zero out the block for other materials
                    local_A = np.zeros_like(self.m)
                # Append to the list of blocks
                blocks.append(local_A)
            # Create the block diagonal matrix for the current material
            A = scipy.sparse.block_diag(blocks, format="csc")
            self.material_mass_matrices.append(A)

    # %%
    def assemble_global_mass_matrix(self, material_properties):
        """
        Assembles the global mass matrix by summing over the list of material-specific
        mass matrices, each weighted by its corresponding material property.

        Parameters:
        - material_mass_matrices: list of sparse matrices (CSC format), one per material.
        - material_properties: dict mapping material IDs to their property values.

        Returns:
        - A: sparse global mass matrix (CSC format).
        """
        # Initialize the global mass matrix as a sparse zero matrix
        global_mass_matrix = scipy.sparse.csc_matrix((self.n_dofs, self.n_dofs))

        # Loop over the material mass matrices and their corresponding properties
        for imat, material_matrix in enumerate(self.material_mass_matrices):
            # Scale the material-specific matrix by its property
            # Add the scaled matrix to the global mass matrix
            global_mass_matrix += material_properties[imat] * material_matrix
        return global_mass_matrix

    # %%
    def compute_source_vectors_by_material(self):
        """
        Constructs a list of global source vectors, one per unique material ID.
        Each vector corresponds to the contributions of the source term for cells
        belonging to the respective material.

        Returns:
        - material_source_vectors: list of NumPy arrays, one per material.
        """
        # Extract unique material IDs
        unique_material_ids = np.unique(self.MSH.cell2mat)

        # List to store source vectors for each material
        self.material_source_vectors = []

        # Outer loop over unique material IDs
        for material_id in unique_material_ids:
            # Initialize the global source vector as a zero array
            global_vector = np.zeros(self.n_dofs)

            for iel in range(self.MSH.n_cells):
                # Check if the current cell belongs to the material
                if self.MSH.cell2mat[iel] == material_id:
                    # Compute the scaled local source vector
                    local_vector = self.f * self.MSH.dx[iel] / 2.0
                    # Global indices for the current element
                    i1 = iel * 2
                    i2 = i1 + 1
                    # Add the local contribution to the global vector
                    global_vector[i1 : i2 + 1] = local_vector[:]
            # Append the global vector to the list
            self.material_source_vectors.append(global_vector)

    # %%
    def assemble_global_qext_vector(self, material_properties):
        """
        Assembles the global source vector by summing over the list of material-specific
        source vectors, each weighted by its corresponding material property.

        Parameters:
        - material_source_vectors: list of vectors, one per material.
        - material_properties: dict mapping material IDs to their property values.

        Returns:
        - global_vector: global qext vector
        """
        # Initialize the global vector
        global_vector = np.zeros(self.n_dofs)

        # Loop over the material mass matrices and their corresponding properties
        for imat, material_source_vector in enumerate(self.material_source_vectors):
            # Scale the material-specific vector by its property
            # Add the scaled vector to the global mass matrix
            global_vector += material_properties[imat] * material_source_vector
        return global_vector

    # %%
    def compute_tot_src(self, phi):
        # only zero-th moment here
        return self.compute_scat_src(phi) + self.qext

    # %%
    def compute_scat_src(self, phi):
        # only zero-th moment here
        return self.M_scattering @ phi

    # %%
    def transport_sweep(self, q, psi_bc):
        # short-cuts:
        n_cells = self.MSH.n_cells
        ndir = self.AQ.ndir

        # local
        mat = np.zeros((2, 2))
        rhs = np.zeros(2)
        # mem alloc
        phi = np.zeros(self.n_dofs)
        # for plotting
        psi = np.zeros((self.n_dofs, ndir))
        # for balance
        psi_exit = np.zeros_like(psi_bc)

        # loop over all directions
        for dir_ in range(ndir):
            # select direction and sweep order
            mu = self.AQ.mu_q[dir_]
            if mu > 0:
                ibeg = 0
                iend = n_cells
                incr = +1
            else:
                ibeg = n_cells - 1
                iend = -1
                incr = -1
            # bc
            psi_in = psi_bc[dir_]

            # loop over cells
            for iel in range(ibeg, iend, incr):
                jac = self.MSH.dx[iel] / 2.0
                mat[:, :] = mu * self.g + self.sigt[iel] * self.m * jac
                i1 = iel * 2
                i2 = i1 + 1
                rhs[:] = q[i1 : i2 + 1]
                if mu > 0:
                    mat[1, 1] += mu
                    rhs[0] += psi_in * mu
                else:
                    mat[0, 0] -= mu
                    rhs[1] -= psi_in * mu
                # cell solve
                psi_cell = np.linalg.solve(mat, rhs)
                # accumulate the scalar flux integral
                phi[i1 : i2 + 1] += psi_cell * self.AQ.w_q[dir_]
                # store cell psi (only for plotting)
                psi[i1 : i2 + 1, dir_] = psi_cell
                # prepare for next cell
                if mu > 0:
                    psi_in = psi_cell[-1]
                else:
                    psi_in = psi_cell[0]
                # last cell in sweep, save outgoing fluxes for balance
                if iel == iend - incr:
                    psi_exit[dir_] = psi_in
        return phi, psi, psi_exit

    # %%
    def action(self, x, op_type="phi"):
        # compute DL^{-1} in two modes

        # if x = phi, out = ( I - D.L^{-1}.M.Scat ) phi
        if op_type == "phi":
            q = self.compute_scat_src(x)
            null_psi_bc = np.zeros_like(self.psi_bc)
            Ax, _, _ = self.transport_sweep(q, null_psi_bc)
            return x - Ax
        # if x = qext, out = D.L^{-1}.M x
        #    when op_type/='phi', volumetric and bc sources are accounted for
        else:
            Ax, _, _ = self.transport_sweep(x, self.psi_bc)
            return Ax

    # %%
    def ksp_solver(self, tolerance=1e-5, restart=1000, verbose=False):
        # solves transport L.Psi = M.Scat.phi + q
        # that is, phi = D.L^{-1}.M.Scat.phi + D.L^{-1}.q
        # as the following Ax=rhs linear system
        # ( I - D.L^{-1}.M.scat ) phi = D.L^{-1}.q

        # declare a counter for printing output
        counter = gmres_counter(verbose)

        # compute rhs = D.L{-1}.M.qext
        rhs = self.action(self.qext, op_type="qext")

        # compute (I - D.L{-1}.M.Scat) Phi = D.L{-1}.M.qext
        # declare the linear operator
        N = len(rhs)
        Ax = LinearOperator((N, N), matvec=self.action)
        Phi = gmres(Ax, rhs, tol=tolerance, callback=counter, restart=restart)[0]

        # one more sweep to also retrieve angular quantities
        qtot = self.compute_tot_src(Phi)
        Phi2, psi_cell, psi_exit = self.transport_sweep(qtot, self.psi_bc)
        if np.linalg.norm(Phi - Phi2) > 10 * tolerance:
            print(
                "Warning: scalar flux changed too much ({}) in last sweep in ksp_solver".format(
                    np.linalg.norm(Phi - Phi2)
                )
            )
        return Phi2, psi_cell, psi_exit

    # %%
    def solve(self, SI_tol=1e-4, SI_max=10, verbose=0):
        # initialize flux moments
        phi = np.zeros(self.n_dofs)

        # compute initial total source
        q_tot = self.compute_tot_src(phi)

        # source iteration loop
        for SI_iter in range(SI_max):
            # save old scalar flux for error convergence
            phi_old = np.copy(phi)
            # perform sweeps
            phi, psi_c, psi_exit = self.transport_sweep(q_tot, self.psi_bc)

            # compute error in successive iterates of the scalar flux
            err_ = np.linalg.norm(phi - phi_old)

            # printout
            if verbose > 0:
                if verbose > 99 or SI_iter % int(pow(10, verbose)) == 0:
                    print("Iteration {0:>5}, error = {1:3.5e}".format(SI_iter, err_))
            # convergence check
            if err_ < SI_tol:
                print("Iteration {0:>5}, error = {1:3.5e}".format(SI_iter, err_))
                print("Converged")
                break
            else:
                # update total source
                q_tot = self.compute_tot_src(phi)
            # warning
            if SI_iter == SI_max - 1:
                print("Iteration {0:>5}, error = {1:3.5e}".format(SI_iter, err_))
                print("Not enough SI")
        return phi, psi_c, psi_exit

    # %%
    def balance(self, phi, psi_exit):
        # volumetric rate integration (self.qext already contains dx)
        SR = np.sum(self.qext)
        # absorption rate integration
        AR = np.sum(self.M_abso @ phi)

        # leakage
        LeakIn, LeakOut = np.zeros(2), np.zeros(2)
        # split directions
        n2 = int(self.AQ.ndir / 2)
        w_neg = self.AQ.w_q[:n2]
        w_pos = self.AQ.w_q[n2:]
        mu_neg = -self.AQ.mu_q[:n2]
        mu_pos = self.AQ.mu_q[n2:]

        # Leak-In Right
        psi_in = self.psi_bc[:n2]
        LeakIn[1] = np.sum(w_neg * mu_neg * psi_in)
        # Leak-Out Left
        psi_out = psi_exit[:n2]
        LeakOut[0] = np.sum(w_neg * mu_neg * psi_out)

        # Leak-In Left
        psi_in = self.psi_bc[n2:]
        LeakIn[0] = np.sum(w_pos * mu_pos * psi_in)
        # Leak-Out Right
        psi_out = psi_exit[n2:]
        LeakOut[1] = np.sum(w_pos * mu_pos * psi_out)

        print("\nParticle Balance:")
        print("\t source rate     = ", SR)
        print("\t absorption rate = ", AR)
        print("\t in-leak left    = ", LeakIn[0])
        print("\t in-leak right   = ", LeakIn[1])
        print("\t out-leak left   = ", LeakOut[0])
        print("\t out-leak right  = ", LeakOut[1])
        print("----")
        gain = SR + LeakIn[0] + LeakIn[1]
        loss = AR + LeakOut[0] + LeakOut[1]
        print("Gain = ", gain)
        print("Loss = ", loss)
        print("Conservation (%) ", np.abs(gain - loss) / gain * 100)

    # %%
    def eddington(self, phi, psi):
        E = np.zeros_like(phi)

        # volumetric rate integration
        for iel in range(self.MSH.n_cells):
            E[iel] = np.sum(psi[iel, :] * self.AQ.w_q * self.AQ.mu_q**2)
        E /= phi
        # plot
        plt.figure()
        plt.plot(self.MSH.xm, E)
        plt.plot(self.MSH.xm, np.ones(self.MSH.n_cells) / 3)
        plt.grid("on", color="black")
        plt.title("Eddington factor")
        plt.show()

    # %%
    def plot_fluxes(
        self,
        phi=None,
        psi=None,
        groups="all",
        directions="all",
        new_fig_handle=True,
        add_legend=False,
        add_title=False,
    ):
        import seaborn as sns
        import itertools as itertools

        # Combine different seaborn palettes
        palette1 = sns.color_palette("husl", n_colors=20)
        palette2 = sns.color_palette("Set2", n_colors=10)
        palette3 = sns.color_palette("tab20", n_colors=10)
        palette4 = sns.color_palette("Accent", n_colors=10)
        # Combine and shuffle them to ensure variety
        palette = palette1 + palette2 + palette3 + palette4
        palette = sorted(palette, key=lambda k: k[0], reverse=True)
        # simple way
        palette = [
            "b",
            "g",
            "r",
            "c",
            "m",
            "y",
            "k",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]

        # --------------------------------------
        # simpler plotter for DG values in 1D
        # --------------------------------------
        def plot_dg(x, y, col, mar="", lab=""):
            nel = len(x) - 1
            for iel in range(nel):
                i1 = iel * 2
                i2 = i1 + 1
                if iel == 0:
                    plt.plot(
                        x[iel : iel + 2],
                        y[i1 : i2 + 1],
                        color=col,
                        marker=mar,
                        label=lab,
                    )
                else:
                    plt.plot(x[iel : iel + 2], y[i1 : i2 + 1], color=col, marker=mar)

        # main routine
        if not isinstance(phi, np.ndarray) and not isinstance(psi, np.ndarray):
            print("nothing to plot, phi or psi must be ndarrays")
            return
        if isinstance(phi, np.ndarray):
            if new_fig_handle:
                plt.figure()
            # Create a cycle iterator for the colors
            color_cycle = itertools.cycle(palette)
            plot_dg(self.MSH.x[:], phi, next(color_cycle), lab="scalar flx")
            plt.grid("on")
            if add_legend:
                plt.legend()
            if add_title:
                plt.title("Flux moments")
            plt.show()
        # psi = (ndofs, ndir)
        if isinstance(psi, np.ndarray):
            ndir = psi.shape[-1]
            if directions == "all":
                directions = np.arange(0, ndir)
            else:
                if isinstance(directions, list):
                    directions = np.array(directions, dtype=int)
                elif not isinstance(directions, np.ndarray):
                    raise Exception("directions must be: 'all' or a list or an ndarray")
                else:
                    directions = directions.astype(int)
            if new_fig_handle:
                plt.figure()
            # Create a cycle iterator for the colors
            color_cycle = itertools.cycle(palette)
            for d in directions:
                plot_dg(
                    self.MSH.x[:],
                    psi[:, d],
                    next(color_cycle),
                    lab="aflx, d" + str(d),
                )
            plt.grid("on")
            if add_legend:
                plt.legend()
            if add_title:
                plt.title("Angular flux")
            plt.show()


# %%
from AQ import AQ
from MESH import MESH

if __name__ == "__main__":
    print("Running FLXSLV as the main code:")

    # select angular quadrature
    myAQ = AQ(4)
    print("ndir = ", myAQ.ndir)

    # create problem mesh
    width = np.array([2.0])
    n_ref = np.array([20], dtype=int)
    myMESH = MESH(width, n_ref)
    print("n_cells = ", myMESH.n_cells)
    print("n_zones = ", myMESH.n_zones)

    # create problem XS
    myXS = {}
    myXS["sigt"] = np.array([50.0])
    myXS["sigs"] = np.array([0.9])

    # create problem SRC terms
    mySRC = {}
    mySRC["qext"] = np.array([1.0])
    mySRC["psi_bc"] = np.zeros(myAQ.ndir)

    # create flux solver object but do not solve anything
    myFLX = FLXSLV(myAQ, myMESH, myXS, mySRC)
