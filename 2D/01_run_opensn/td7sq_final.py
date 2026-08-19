#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 2D Transport test

import os
import sys
import numpy as np

if "opensn_console" not in globals():
    from mpi4py import MPI
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    from pyopensn.mesh import OrthogonalMeshGenerator
    from pyopensn.xs import MultiGroupXS
    from pyopensn.source import VolumetricSource
    from pyopensn.aquad import GLCProductQuadrature2DXY
    from pyopensn.solver import DiscreteOrdinatesProblem, SteadyStateSolver
    from pyopensn.fieldfunc import FieldFunctionInterpolationVolume
    from pyopensn.logvol import RPPLogicalVolume


def ramp_q(time_value, q0, t_ramp):
    if time_value <= 0.0:
        return 0.0
    if time_value < t_ramp:
        return q0 * time_value / t_ramp
    return q0

if __name__ == "__main__":

    #meshgen = FromFileMeshGenerator(filename="two_squares.msh")
    meshgen = FromFileMeshGenerator(filename="square.msh")
    grid = meshgen.Execute()
    grid.SetOrthogonalBoundaries()
    grid.ExportToPVTU("Read_gmsh")

    num_groups = 1
    vel = 5.0
    # 12 = central block
    # 13 = background
    # 1-11= all other squares
    # corners = 1,2,9,11

    # other blocks
    xs_fis = MultiGroupXS()
    xs_fis.LoadFromOpenSn(os.path.join(os.path.dirname(__file__), "xs_fis_vel_src2.xs"))
    # central src
    xs_src = MultiGroupXS()
    xs_src.CreateSimpleOneGroup(sigma_t=1.0, c=0.75, velocity=vel)
    # other/abso
    xs_other = MultiGroupXS()
    xs_other.CreateSimpleOneGroup(sigma_t=1., c=0.0, velocity=vel)
    # background
    xs_bkg = MultiGroupXS()
    xs_bkg.CreateSimpleOneGroup(sigma_t=0.0, c=0.0, velocity=vel)

    q0 = 0.0
    t_ramp = 0. 
    def source_func(group, time_value):
        return ramp_q(time_value, q0, t_ramp)
    vol_src = VolumetricSource(block_ids=[12], strength_function=source_func)

    # Setup Physics
    pquad = GLCProductQuadrature2DXY(n_polar=2, n_azimuthal=32, scattering_order=0)

    num_angles = len(pquad.omegas)
    lst_angle_ids = list(range(num_angles))
    if rank == 0:
        print('number of directions =',num_angles)

    angle_mask = np.zeros(num_angles, dtype=bool)
    angle_mask[num_angles-1] = True
    # Return a constant inflow on xmin for the masked angles and all groups
    def bc_func(group_index, angle_index):
        if angle_mask[angle_index]:
            return 1.0 
        return 0.0

    xmin_bc = AngularFluxFunction(bc_func)

    phys = DiscreteOrdinatesProblem(
        mesh=grid,
        num_groups=num_groups,
        groupsets=[
            {
                "groups_from_to": [0, 0],
                "angular_quadrature": pquad,
                "angle_aggregation_num_subsets": 1,
                "inner_linear_method": "petsc_gmres",
                "l_abs_tol": 1.0e-8,
                "l_max_its": 300,
                "gmres_restart_interval": 100,
            },
        ],
        xs_map=[
            {"block_ids": [ 1, 2,4, 7, 10, 11 ], "xs": xs_fis},
            {"block_ids": [ 3,5,6,8,9 ], "xs": xs_other},
            {"block_ids": [ 12 ], "xs": xs_src},
            {"block_ids": [ 13 ], "xs": xs_bkg},
        ],
#        volumetric_sources= [vol_src],
         boundary_conditions=[
            {
                "name": "ymax",
                "type": "isotropic",
                "group_strength": [0.],
            },
            {
                "name": "xmin",
                "type": "arbitrary",
                "function": xmin_bc,
            },
            {
                "name": "xmax",
                "type": "isotropic",
                "group_strength": [0.],
            },
        ],
        options = {"save_angular_flux": True},
    )

    if False:
        keigen = PowerIterationKEigenSolver(problem=phys, max_iters=200, k_tol=1.0e-10)
        keigen.Initialize()
        keigen.Execute()
        keff = keigen.GetEigenvalue()
        raise ValueError(f'done, {keff}')

    solver = TransientSolver(problem=phys, initial_state="zero")
    solver.Initialize()
    solver.SetTheta(0.5)

    my_dt = 0.005
    stop_time = 5.0
    current_time = 0.0

    vtk_root = "flux_3newh"
    vtk_roota = "aflux_3newh"

    save_every = 1        # e.g. N = 10
    iter_step = 0           # counts all time steps taken
    step = 0                # counts saved frames (used for filenames)

    while current_time < stop_time:
        target_time = min(current_time + my_dt, stop_time)
        solver.SetTimeStep(target_time - current_time)
        solver.Advance()
        current_time = target_time
        iter_step += 1

        # save every N steps, and also force-save the final state at stop_time
        do_save = (iter_step % save_every == 0) or (current_time >= stop_time)

        if do_save:
            step += 1

            # --- export ---
            # fflist = phys.GetFieldFunctions()
            fflist = phys.GetScalarFieldFunctionList()
            afflist = phys.GetAngularFieldFunctionList(groups=[0], angles=lst_angle_ids)

            vtk_basename = f"{vtk_root}_{step:04d}"   # -> flux_3newh_0001, flux_3newh_0002, ...
            FieldFunctionGridBased.ExportMultipleToPVTU(fflist, vtk_basename)

            vtk_basename = f"{vtk_roota}_{step:04d}"   # -> aflux_3newh_0001, aflux_3newh_0002, ...
            FieldFunctionGridBased.ExportMultipleToPVTU(afflist, vtk_basename)


    # phys.WriteAngularFluxes("flux")
