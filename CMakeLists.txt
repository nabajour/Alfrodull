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

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
#if (CUDA_FOUND AND SM MATCHES "0")
#  CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
#else()
#  message(STATUS "CUDA Architecture manually set to: -arch=sm_${SM}")
#  set(ARCH_FLAGS "-arch=sm_${SM}")
#endif()

SET(CUDA_SEPARABLE_COMPILATION ON)

# build flags
LIST(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
LIST(APPEND CUDA_NVCC_FLAGS ${COMPILE_FLAGS})
LIST(APPEND CUDA_NVCC_FLAGS "-std c++11")

string (APPEND CMAKE_CUDA_FLAGS " -dc -cudart shared -rdc=true -lcudart -lcudadevrt -lcuda" )

find_package(pybind11 REQUIRED)
#find_package(PythonLibs REQUIRED)


set(SRC_PYMODULE
	pylfrodull/src/pylfrodull.cpp)

set(SRC_ALFRODULLIB
	src/alfrodullib.cu
	src/kernels/atomic_add.cu
	src/kernels/integrate_flux.cu)
	
include_directories(src/inc)
include_directories(pylfrodull/src/inc)
#include_directories(${PYTHON_INCLUDE_DIRS})
# generate python submodule
# add_subdirectory(pylfrodull/src)

#add_library(alfrodullib SHARED ${SRC_ALFRODULLIB})
#OPTIONS -DBUILD_LEVEL="release" -DDEVICE_SM=${SM})

##pybind11 sets -fvisibility=hidden in INTERFACE_COMPILE_OPTIONS on it's module target
get_target_property(modifacecopts pybind11::module INTERFACE_COMPILE_OPTIONS)
list(REMOVE_ITEM modifacecopts "-fvisibility=hidden")
set_target_properties(pybind11::module PROPERTIES INTERFACE_COMPILE_OPTIONS "${modifacecopts}")

add_library(pylfrodull SHARED ${SRC_PYMODULE} ${SRC_ALFRODULLIB})
#target_link_libraries (pylfrodull alfrodullib)

target_link_libraries(pylfrodull PRIVATE pybind11::module ${CUDA_LIBRARIES})
#target_link_libraries(pylfrodull PRIVATE ${PYTHON_LIBRARIES} ${CUDA_LIBRARIES})
set_target_properties(pylfrodull PROPERTIES
				 POSITION_INDEPENDENT_CODE ON
				 CUDA_VISIBILITY_PRESET "hidden"
				 CXX_VISIBILITY_PRESET "hidden"
				 PREFIX "${PYTHON_MODULE_PREFIX}"
                                 SUFFIX "${PYTHON_MODULE_EXTENSION}")
