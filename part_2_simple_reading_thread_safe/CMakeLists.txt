cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(part_1_simple_reading_thread_safe	LANGUAGES	CUDA	CXX)

find_package(CUDA REQUIRED)
add_compile_options(-std=c++11)
enable_testing()

set(CUDA_NVCC_FLAGS "-std=c++11")

set(CMAKE_CXX_STANDARD	    11)
set(CMAKE_CUDA_STANDARD	    11)
set(CMAKE_CXX_EXTENSIONS	Off)
set(CMAKE_CUDA_EXTENSIONS	Off)

set(CMAKE_CUDA_FLAGS    "${CMAKE_CUDA_FLAGS}   -arch=compute_75 -code=sm_75")
CUDA_ADD_LIBRARY(${PROJECT_NAME}_cuda_lib
        src/analyzer.cu
        inc/analyzer.cuh)

add_executable(${PROJECT_NAME} main.cpp inc/ThreadSafeQueue.h inc/helpers.h src/helpers.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_cuda_lib)
