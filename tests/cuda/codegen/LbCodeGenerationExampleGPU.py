import sympy as sp
import pystencils as ps
from lbmpy.stencils import get_stencil
from lbmpy.creationfunctions import create_lb_method, create_lb_collision_rule, create_lb_update_rule
from lbmpy.boundaries import NoSlip, UBB
from lbmpy.macroscopic_value_kernels import macroscopic_values_setter
from pystencils_walberla import CodeGeneration, generate_sweep, generate_pack_info_from_kernel
from lbmpy_walberla import RefinementScaling, generate_boundary, generate_lattice_model

with CodeGeneration() as ctx:
    stencil = get_stencil("D3Q19")

    dim = len(stencil[0])
    q = len(stencil)

    omega = sp.symbols("omega")
    omega_free = 1
    vel_field = ps.fields(f"velocity({dim}): [3D]", layout='zyxf')
    pdfs = ps.fields(f"pdfs({q}): [3D]", layout='zyxf')

    lb_method = create_lb_method(stencil=stencil, compressible=True, method='mrt',
                                 relaxation_rates=[omega, omega, omega_free, omega_free, omega_free, omega_free])

    # the collision rule of the LB method
    collision_rule = create_lb_collision_rule(
        lb_method=lb_method,
        output={'velocity': vel_field},
        optimization={'cse_global': True}
    )

    update_rule = create_lb_update_rule(collision_rule=collision_rule)

    setter_assignments = macroscopic_values_setter(lb_method, velocity=[0]*dim, pdfs=pdfs.center_vector, density=1)

    # generate lattice model and (optionally) boundary conditions
    # for CPU simulations waLBerla's internal boundary handling can be used as well
    generate_lattice_model(ctx, 'LbCodeGenerationExample_LatticeModel', collision_rule, target='gpu')
    generate_sweep(ctx, 'PDF_Setter', setter_assignments, target='gpu')
    generate_boundary(ctx, 'LbCodeGenerationExample_UBB', UBB([0.05, 0, 0]), collision_rule.method, target='gpu')
    generate_boundary(ctx, 'LbCodeGenerationExample_NoSlip', NoSlip(), collision_rule.method, target='gpu')

    generate_pack_info_from_kernel(ctx, "PackInfo", update_rule, target='gpu')
