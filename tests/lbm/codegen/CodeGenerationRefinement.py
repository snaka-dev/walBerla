import sympy as sp
import pystencils as ps
from lbmpy.creationfunctions import create_lb_collision_rule
from lbmpy.boundaries import NoSlip, UBB, FixedDensity
from pystencils_walberla import CodeGeneration
from lbmpy_walberla import RefinementScaling, generate_boundary, generate_lattice_model

omega = sp.symbols("omega")
omega_free = sp.Symbol("omega_free")
omega_fill = sp.symbols("omega_:10")

options_dict = {
    'D3Q19_SRT_INCOMP': {
        'method': 'srt',
        'stencil': 'D3Q19',
        'relaxation_rates': [omega],
        'compressible': False,
    },
    'D3Q19_SRT_COMP': {
        'method': 'srt',
        'stencil': 'D3Q19',
        'relaxation_rates': [omega],
        'compressible': True,
    },
    'D3Q27_SRT_INCOMP': {
        'method': 'srt',
        'stencil': 'D3Q27',
        'relaxation_rates': [omega],
        'compressible': False,
    },
    'D3Q27_SRT_COMP': {
        'method': 'srt',
        'stencil': 'D3Q27',
        'relaxation_rates': [omega],
        'compressible': True,
    },
    'D3Q27_TRT_INCOMP': {
        'method': 'trt',
        'stencil': 'D3Q27',
        'relaxation_rates': [omega, omega],
        'compressible': False,
    },
    'D3Q19_MRT_COMP': {
        'method': 'mrt',
        'stencil': 'D3Q27',
        'relaxation_rates': [omega, omega, omega, omega, omega, omega],
        'compressible': True,
    }
}

with CodeGeneration() as ctx:

    config_name = ctx.config

    collision_rule = create_lb_collision_rule(
        stencil=options_dict[config_name]['stencil'],
        compressible=options_dict[config_name]['compressible'],
        method=options_dict[config_name]['method'],
        relaxation_rates=options_dict[config_name]['relaxation_rates'],
        optimization={'cse_global': True}
    )

    scaling = RefinementScaling()
    scaling.add_standard_relaxation_rate_scaling(omega)

    generate_lattice_model(ctx, 'CodeGenerationRefinement_' + config_name + '_LatticeModel', collision_rule, refinement_scaling=scaling)
