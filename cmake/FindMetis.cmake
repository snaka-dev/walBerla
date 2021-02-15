#[=======================================================================[.rst:
FindMETIS
-------

Finds the METIS library.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``METIS_FOUND``
  True if the system has the METIS library.
``METIS_INCLUDE_DIRS``
  Include directories needed to use METIS.
``METIS_LIBRARIES``
  Libraries needed to link to METIS.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``METIS_INCLUDE_DIR``
  The directory containing ``METIS.h``.
``METIS_LIBRARY``
  The path to the METIS library.

#]=======================================================================]

set(CMAKE_FIND_DEBUG_MODE TRUE)
find_path(METIS_INCLUDE_DIR
        NAMES metis.h
        )
     set(CMAKE_FIND_DEBUG_MODE FALSE)
find_library(METIS_LIBRARY
        NAMES metis
        )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Metis
        FOUND_VAR METIS_FOUND
        REQUIRED_VARS
        METIS_LIBRARY
        METIS_INCLUDE_DIR
        VERSION_VAR METIS_VERSION
        )

if(METIS_FOUND)
    # Build and run test program
    try_compile(METIS_TEST_RUNS 
       "${CMAKE_BINARY_DIR}" 
       "${CMAKE_SOURCE_DIR}/cmake/TestMetis.cpp"
       CXX_STANDARD 14
       OUTPUT_VARIABLE METIS_ERROR
       CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${METIS_INCLUDE_DIR}")

    if (NOT METIS_TEST_RUNS)
       message(FATAL_ERROR "Metis not build with 64-bit integer and 64-bit float. (${METIS_INCLUDE_DIR})")
    endif()
    set(METIS_LIBRARIES ${METIS_LIBRARY})
    set(METIS_INCLUDE_DIRS ${METIS_INCLUDE_DIR})
endif()

mark_as_advanced(
        METIS_INCLUDE_DIR
        METIS_LIBRARY
)
