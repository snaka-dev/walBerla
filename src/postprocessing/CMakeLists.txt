###################################################################################################
#
# Module postprocessing
#
###################################################################################################

add_library( postprocessing )
target_link_libraries( postprocessing PUBLIC core domain_decomposition field python_coupling stencil geometry )
target_sources( postprocessing
      PRIVATE
      MarchingCubes.impl.h
      MarchingCubes.h
      FieldToSurfaceMesh.h
      FieldToSurfaceMesh.cpp
      )

###################################################################################################
