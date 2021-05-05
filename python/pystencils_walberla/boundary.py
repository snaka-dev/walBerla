import numpy as np
from jinja2 import Environment, PackageLoader, StrictUndefined
from pystencils import Field, FieldType
from pystencils.boundaries.boundaryhandling import create_boundary_kernel
from pystencils.boundaries.createindexlist import (
    boundary_index_array_coordinate_names, direction_member_name,
    numpy_data_type_for_boundary_object)
from pystencils.data_types import TypedSymbol, create_type
from pystencils_walberla.codegen import default_create_kernel_parameters
from pystencils_walberla.jinja_filters import add_pystencils_filters_to_jinja_env
from pystencils_walberla.additional_data_handler import AdditionalDataHandler
from pystencils_walberla.kernel_selection import (
    KernelFamily, AbstractKernelSelectionNode, KernelCallNode, HighLevelInterfaceSpec)
from pystencils.astnodes import KernelFunction


def generate_boundary(generation_context,
                      class_name,
                      boundary_object,
                      field_name,
                      neighbor_stencil,
                      index_shape,
                      field_type=FieldType.GENERIC,
                      kernel_creation_function=None,
                      target='cpu',
                      namespace='pystencils',
                      additional_data_handler=None,
                      interface_mappings=(),
                      generate_functor=True,
                      **create_kernel_params):

    if boundary_object.additional_data and additional_data_handler is None:
        raise ValueError("Boundary object has additional data but you have not provided an AdditionalDataHandler.")

    struct_name = "IndexInfo"
    boundary_object.name = class_name
    dim = len(neighbor_stencil[0])

    create_kernel_params = default_create_kernel_parameters(generation_context, create_kernel_params)
    create_kernel_params["target"] = target
    del create_kernel_params["cpu_vectorize_info"]

    if not create_kernel_params["data_type"]:
        create_kernel_params["data_type"] = 'double' if generation_context.double_accuracy else 'float32'
    index_struct_dtype = numpy_data_type_for_boundary_object(boundary_object, dim)

    field = Field.create_generic(field_name, dim,
                                 np.float64 if generation_context.double_accuracy else np.float32,
                                 index_dimensions=len(index_shape), layout='fzyx', index_shape=index_shape,
                                 field_type=field_type)

    index_field = Field('indexVector', FieldType.INDEXED, index_struct_dtype, layout=[0],
                        shape=(TypedSymbol("indexVectorSize", create_type(np.int64)), 1), strides=(1, 1))

    if not kernel_creation_function:
        kernel_creation_function = create_boundary_kernel

    kernel = kernel_creation_function(field, index_field, neighbor_stencil, boundary_object, **create_kernel_params)

    if isinstance(kernel, KernelFunction):
        kernel.function_name = "boundary_" + boundary_object.name
        selection_tree = KernelCallNode(kernel)
    elif isinstance(kernel, AbstractKernelSelectionNode):
        selection_tree = kernel
    else:
        raise ValueError(f"kernel_creation_function returned wrong type: {kernel.__class__}")

    kernel_family = KernelFamily(selection_tree, class_name)
    interface_spec = HighLevelInterfaceSpec(kernel_family.kernel_selection_parameters, interface_mappings)

    if additional_data_handler is None:
        additional_data_handler = AdditionalDataHandler(stencil=neighbor_stencil)

    context = {
        'kernel': kernel_family,
        'class_name': boundary_object.name,
        'interface_spec': interface_spec,
        'generate_functor': generate_functor,
        'StructName': struct_name,
        'StructDeclaration': struct_from_numpy_dtype(struct_name, index_struct_dtype),
        'dim': dim,
        'target': target,
        'namespace': namespace,
        'inner_or_boundary': boundary_object.inner_or_boundary,
        'additional_data_handler': additional_data_handler
    }

    env = Environment(loader=PackageLoader('pystencils_walberla'), undefined=StrictUndefined)
    add_pystencils_filters_to_jinja_env(env)

    header = env.get_template('Boundary.tmpl.h').render(**context)
    source = env.get_template('Boundary.tmpl.cpp').render(**context)

    source_extension = "cpp" if target == "cpu" else "cu"
    generation_context.write_file(f"{class_name}.h", header)
    generation_context.write_file(f"{class_name}.{source_extension}", source)


def generate_staggered_boundary(generation_context, class_name, boundary_object,
                                dim, neighbor_stencil, index_shape, target='cpu', **kwargs):
    assert dim == len(neighbor_stencil[0])
    generate_boundary(generation_context, class_name, boundary_object, 'field', neighbor_stencil, index_shape,
                      FieldType.STAGGERED, target=target, **kwargs)


def generate_staggered_flux_boundary(generation_context, class_name, boundary_object,
                                     dim, neighbor_stencil, index_shape, target='cpu', **kwargs):
    assert dim == len(neighbor_stencil[0])
    generate_boundary(generation_context, class_name, boundary_object, 'flux', neighbor_stencil, index_shape,
                      FieldType.STAGGERED_FLUX, target=target, **kwargs)


def struct_from_numpy_dtype(struct_name, numpy_dtype):
    result = f"struct {struct_name} {{ \n"

    equality_compare = []
    constructor_params = []
    constructor_initializer_list = []
    for name, (sub_type, offset) in numpy_dtype.fields.items():
        pystencils_type = create_type(sub_type)
        result += f"    {pystencils_type} {name};\n"
        if name in boundary_index_array_coordinate_names or name == direction_member_name:
            constructor_params.append(f"{pystencils_type} {name}_")
            constructor_initializer_list.append(f"{name}({name}_)")
        else:
            constructor_initializer_list.append(f"{name}()")
        if pystencils_type.is_float():
            equality_compare.append(f"floatIsEqual({name}, o.{name})")
        else:
            equality_compare.append(f"{name} == o.{name}")

    result += "    %s(%s) : %s {}\n" % \
              (struct_name, ", ".join(constructor_params), ", ".join(constructor_initializer_list))
    result += "    bool operator==(const %s & o) const {\n        return %s;\n    }\n" % \
              (struct_name, " && ".join(equality_compare))
    result += "};\n"
    return result
