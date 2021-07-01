waLBerla_link_files_to_builddir( *.prm )

waLBerla_add_executable ( NAME walberlaSRT
                          FILES walberlaSRT.cpp
                          DEPENDS blockforest core field lbm geometry timeloop gui )

if( WALBERLA_BUILD_WITH_CODEGEN )
    walberla_generate_target_from_python( NAME srtLatticeModel
                                          FILE latticeModel.py
                                          OUT_FILES   SRTLatticeModel.cpp SRTLatticeModel.h )

    waLBerla_add_executable ( NAME lbmpySRT
                              FILES lbmpySRT.cpp
                              DEPENDS blockforest core field lbm geometry timeloop gui srtLatticeModel)
endif()