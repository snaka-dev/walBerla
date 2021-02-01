#[=======================================================================[.rst:
FindFFTW3_MPI
-------

Finds the FFTW3_MPI library.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``FFTW3_MPI_FOUND``
  True if the system has the FFTW3_MPI library.
``FFTW3_MPI_INCLUDE_DIRS``
  Include directories needed to use FFTW3_MPI.
``FFTW3_MPI_LIBRARIES``
  Libraries needed to link to FFTW3_MPI.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``FFTW3_MPI_INCLUDE_DIR``
  The directory containing ``FFTW3_MPI.h``.
``FFTW3_MPI_LIBRARY``
  The path to the FFTW3_MPI library.

#]=======================================================================]

find_path(FFTW3_MPI_INCLUDE_DIR
  NAMES fftw3-mpi.h
)
find_library(FFTW3_MPI_LIBRARY
  NAMES fftw3_mpi
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW3_MPI
  FOUND_VAR FFTW3_MPI_FOUND
  REQUIRED_VARS
    FFTW3_MPI_LIBRARY
    FFTW3_MPI_INCLUDE_DIR
  VERSION_VAR FFTW3_MPI_VERSION
)

if(FFTW3_MPI_FOUND)
  set(FFTW3_MPI_LIBRARIES ${FFTW3_MPI_LIBRARY})
  set(FFTW3_MPI_INCLUDE_DIRS ${FFTW3_MPI_INCLUDE_DIR})
endif()

mark_as_advanced(
  FFTW3_MPI_INCLUDE_DIR
  FFTW3_MPI_LIBRARY
)

