###################################################################################################
#
# Module executiontree
#
###################################################################################################

add_library( executiontree INTERFACE)
target_link_libraries( executiontree INTERFACE core domain_decomposition timeloop )
target_sources( executiontree
      INTERFACE
      ExecutionTreeSweep.h
      ExecutionTree.cpp
      ExecutionTree.h
      ExecutionTree.impl.h
      )
                        
###################################################################################################                        