#include "stencil/Directions.h"
#include "core/cell/CellInterval.h"
#include "core/DataTypes.h"
#include "{{class_name}}.h"

#if ( defined WALBERLA_CXX_COMPILER_IS_GNU ) || ( defined WALBERLA_CXX_COMPILER_IS_CLANG )
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wfloat-equal"
#   pragma GCC diagnostic ignored "-Wshadow"
#   pragma GCC diagnostic ignored "-Wconversion"
#   pragma GCC diagnostic ignored "-Wunused-variable"
#endif

{% for header in headers %}
#include {{header}}
{% endfor %}

namespace walberla {
namespace {{namespace}} {

using walberla::cell::CellInterval;
using walberla::stencil::Direction;


{% for kernel in pack_kernels.values() %}
{{kernel|generate_definition(target)}}
{% endfor %}

{% for kernel in unpack_kernels.values() %}
{{kernel|generate_definition(target)}}
{% endfor %}


void {{class_name}}::pack(Direction dir, unsigned char * byte_buffer, IBlock * block) const
{
    {{dtype}} * buffer = reinterpret_cast<{{dtype}}*>(byte_buffer);

    {{fused_kernel|generate_block_data_to_field_extraction(parameters_to_ignore=['buffer'])|indent(4)}}
    CellInterval ci;
    {% if gl_to_inner -%}
    {{field_name}}->getGhostRegion(dir, ci, 1, false);
    {%- else -%}
    {{field_name}}->getSliceBeforeGhostLayer(dir, ci, 1, false);
    {%- endif %}

    switch( dir )
    {
        {%- for direction_set, kernel in pack_kernels.items()  %}
        {%- for dir in direction_set %}
        case stencil::{{dir}}:
        {%- endfor %}
        {
            {{kernel|generate_call(cell_interval="ci")|indent(12)}}
            break;
        }
        {% endfor %}

        default:
            WALBERLA_ASSERT(false);
    }
}


void {{class_name}}::unpack(Direction dir, unsigned char * byte_buffer, IBlock * block) const
{
    {{dtype}} * buffer = reinterpret_cast<{{dtype}}*>(byte_buffer);

    {{fused_kernel|generate_block_data_to_field_extraction(parameters_to_ignore=['buffer'])|indent(4)}}
    CellInterval ci;
    {% if gl_to_inner -%}
    {{field_name}}->getSliceBeforeGhostLayer(dir, ci, 1, false);
    {%- else -%}
    {{field_name}}->getGhostRegion(dir, ci, 1, false);
    {%- endif %}
    auto communciationDirection = stencil::inverseDir[dir];

    switch( communciationDirection )
    {
        {%- for direction_set, kernel in unpack_kernels.items()  %}
        {%- for dir in direction_set %}
        case stencil::{{dir}}:
        {%- endfor %}
        {
            {{kernel|generate_call(cell_interval="ci")|indent(12)}}
            break;
        }
        {% endfor %}

        default:
            WALBERLA_ASSERT(false);
    }
}


uint_t {{class_name}}::size(stencil::Direction dir, const IBlock * block) const
{
    {{fused_kernel|generate_block_data_to_field_extraction(parameters_to_ignore=['buffer'])|indent(4)}}
    CellInterval ci;
    {{field_name}}->getGhostRegion(dir, ci, 1, false);

    uint_t elementsPerCell = 0;

    switch( dir )
    {
        {%- for direction_set, elements in elements_per_cell.items()  %}
        {%- for dir in direction_set %}
        case stencil::{{dir}}:
        {%- endfor %}
            elementsPerCell = {{elements}};
            break;
        {% endfor %}
        default:
            elementsPerCell = 0;
    }
    return ci.numCells() * elementsPerCell * sizeof( {{dtype}} );
}



} // namespace {{namespace}}
} // namespace walberla