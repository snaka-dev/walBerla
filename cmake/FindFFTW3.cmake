#[=======================================================================[.rst:
FindFFTW3
-------

Finds the FFTW3 library.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``FFTW3_FOUND``
  True if the system has the FFTW3 library.
``FFTW3_INCLUDE_DIRS``
  Include directories needed to use FFTW3.
``FFTW3_LIBRARIES``
  Libraries needed to link to FFTW3.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``FFTW3_INCLUDE_DIR``
  The directory containing ``FFTW3.h``.
``FFTW3_LIBRARY``
  The path to the FFTW3 library.

#]=======================================================================]

find_path(FFTW3_INCLUDE_DIR
  NAMES fftw3.h
)
find_library(FFTW3_LIBRARY
  NAMES fftw3
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW3
  FOUND_VAR FFTW3_FOUND
  REQUIRED_VARS
    FFTW3_LIBRARY
    FFTW3_INCLUDE_DIR
  VERSION_VAR FFTW3_VERSION
)

if(FFTW3_FOUND)
  set(FFTW3_LIBRARIES ${FFTW3_LIBRARY})
  set(FFTW3_INCLUDE_DIRS ${FFTW3_INCLUDE_DIR})
endif()

mark_as_advanced(
  FFTW3_INCLUDE_DIR
  FFTW3_LIBRARY
)
