cmake_minimum_required (VERSION 3.8 FATAL_ERROR)

# check command line args, user can specify SM to override manual setting
set(SM "0" CACHE STRING "GPU SM value")

# if CUDA wants specific gcc/g++ 
#SET(CMAKE_CXX_COMPILER g++-8)
#SET(CMAKE_C_COMPILER gcc-8)


project (Alfrodull CUDA CXX)
# some C++ flags
set (CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set ( COMPILE_FLAGS --compiler-options;-Wall)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(THOR_BASE_FOLDER "../")


include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
#find_package(CUDA REQUIRED)
#if (CUDA_FOUND AND SM MATCHES "0")
#  CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
#else()
#  message(STATUS "CUDA Architecture manually set to: -arch=sm_${SM}")
#  set(ARCH_FLAGS "-arch=sm_${SM}")
#endif()

#SET(CUDA_SEPARABLE_COMPILATION ON)

# build flags
#LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
#LIST(APPEND CUDA_NVCC_FLAGS ${COMPILE_FLAGS})
#LIST(APPEND CUDA_NVCC_FLAGS "-std c++11")

#string (APPEND CMAKE_CUDA_FLAGS " -dc -cudart shared -rdc=true -lcudart -lcudadevrt -lcuda" )

find_package(pybind11 REQUIRED)
#find_package(PythonLibs REQUIRED)


######################################################################
# FIND HDF5
find_package(HDF5 REQUIRED COMPONENTS CXX)
if(HDF5_FOUND)
        include_directories(${HDF5_INCLUDE_DIR})
	set(HDF5_LIBS hdf5 hdf5_cpp)
endif()

message(STATUS "HDF5 include dir: " ${HDF5_INCLUDE_DIR})
message(STATUS "HDF5 C++libraries: " ${HDF5_LIBRARIES})
######################################################################

set(SRC_PYMODULE
	pylfrodull/src/pylfrodull.cpp)

set(SRC_ALFRODULLIB
	src/alfrodullib.cu
	src/alfrodull_engine.cu
	src/planck_table.cu
	src/opacities/opacities.cu
	src/kernels/calculate_physics.cu
	src/kernels/math_helpers.cu
	src/kernels/interpolate_values.cu
	src/kernels/surface_planck.cu
	src/kernels/correct_surface_emission.cu
	src/kernels/atomic_add.cu
	src/kernels/integrate_flux.cu)

set(SRC_THOR_UTILS
	${THOR_BASE_FOLDER}/src/utils/storage.cpp)
	
include_directories(src/inc)
include_directories(pylfrodull/src/inc)

include_directories(${THOR_BASE_FOLDER}/src/headers/ )


######################################################################
# build and link binaries
add_library(alfrodull STATIC ${SRC_ALFRODULLIB} ${SRC_THOR_UTILS})
set_target_properties(alfrodull PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    CUDA_SEPARABLE_COMPILATION ON
)


add_library(pylfrodull SHARED ${SRC_PYMODULE})
#target_link_libraries (pylfrodull alfrodullib)

set_target_properties(pylfrodull PROPERTIES 
				     CXX_VISIBILITY_PRESET "hidden"
				         INTERPROCEDURAL_OPTIMIZATION TRUE
					     PREFIX "${PYTHON_MODULE_PREFIX}"
					         SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

target_link_libraries(pylfrodull PRIVATE alfrodull)
target_link_libraries(pylfrodull PRIVATE ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES})

target_link_libraries(pylfrodull PRIVATE pybind11::module )
#target_link_libraries(pylfrodull PRIVATE ${PYTHON_LIBRARIES} ${CUDA_LIBRARIES})
# set_target_properties(pylfrodull PROPERTIES
# 				 POSITION_INDEPENDENT_CODE ON
# #				 CUDA_VISIBILITY_PRESET "hidden"
# 				 CXX_VISIBILITY_PRESET "hidden"
# 				 PREFIX "${PYTHON_MODULE_PREFIX}"
#                                  SUFFIX "${PYTHON_MODULE_EXTENSION}")
