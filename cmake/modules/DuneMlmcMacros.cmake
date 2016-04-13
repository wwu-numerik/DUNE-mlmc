# File for module specific CMake tests.

# adding the respective flags would already have happened
# in dune-multiscale. This is only to ensure FFTW was 
# actually found there (where it is still optional if
# multiscale is used solo)
find_package(FFTW REQUIRED)
