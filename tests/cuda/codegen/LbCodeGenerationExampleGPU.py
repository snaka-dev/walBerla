import sympy as sp
import pystencils as ps
from lbmpy.creationfunctions import create_lb_collision_rule
from lbmpy.boundaries import NoSlip, UBB
from pystencils_walberla import CodeGeneration
from lbmpy_walberla import RefinementScaling, generate_boundary, generate_lattice_model

with CodeGeneration() as ctx:
    omega, omega_free = sp.symbols("omega, omega_free")
    force_field, vel_field, omega_out = ps.fields("force(3), velocity(3), omega_out: [3D]", layout='zyxf')

    # the collision rule of the LB method where the some advanced features
    collision_rule = create_lb_collision_rule(
        stencil='D3Q19', compressible=True,
        method='mrt', relaxation_rates=[omega, omega, omega_free, omega_free, omega_free, omega_free],
        optimization={'cse_global': True}
    )

    # generate lattice model and (optionally) boundary conditions
    # for CPU simulations waLBerla's internal boundary handling can be used as well
    generate_lattice_model(ctx, 'LbCodeGenerationExample_LatticeModel', collision_rule, target='gpu')
    generate_boundary(ctx, 'LbCodeGenerationExample_UBB', UBB([0.05, 0, 0]), collision_rule.method, target='gpu')
    generate_boundary(ctx, 'LbCodeGenerationExample_NoSlip', NoSlip(), collision_rule.method, target='gpu')
