cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)

project(genome_analyzer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

find_package(Threads REQUIRED)

add_subdirectory(common_parts)
include_directories(common_parts)

add_subdirectory(data_generator)
add_subdirectory(solutions)