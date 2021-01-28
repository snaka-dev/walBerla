#[=======================================================================[.rst:
FindPFFT
-------

Finds the PFFT library.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``PFFT_FOUND``
  True if the system has the PFFT library.
``PFFT_INCLUDE_DIRS``
  Include directories needed to use PFFT.
``PFFT_LIBRARIES``
  Libraries needed to link to PFFT.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``PFFT_INCLUDE_DIR``
  The directory containing ``pfft.h``.
``PFFT_LIBRARY``
  The path to the PFFT library.

#]=======================================================================]

find_package(PkgConfig)
pkg_check_modules(PC_PFFT QUIET pfft)

find_path(PFFT_INCLUDE_DIR
  NAMES pfft.h
  PATHS ${PC_PFFT_INCLUDE_DIRS}
)
find_library(PFFT_LIBRARY
  NAMES pfft
  PATHS ${PC_PFFT_LIBRARY_DIRS}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PFFT
  FOUND_VAR PFFT_FOUND
  REQUIRED_VARS
    PFFT_LIBRARY
    PFFT_INCLUDE_DIR
  VERSION_VAR PFFT_VERSION
)

if(PFFT_FOUND)
  set(PFFT_LIBRARIES ${PFFT_LIBRARY})
  set(PFFT_INCLUDE_DIRS ${PFFT_INCLUDE_DIR})
endif()

mark_as_advanced(
  PFFT_INCLUDE_DIR
  PFFT_LIBRARY
)
