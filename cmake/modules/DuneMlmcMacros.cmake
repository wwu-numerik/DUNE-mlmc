# File for module specific CMake tests.
find_package(FFTW REQUIRED)
if(FFTW_FOUND)
	include_directories(${FFTW_INCLUDES})
	set(HAVE_RANDOM_PROBLEM 1)
	set(COMMON_LIBS ${COMMON_LIBS} fftw3_mpi ${FFTW_LIBRARIES})
endif(FFTW_FOUND)
