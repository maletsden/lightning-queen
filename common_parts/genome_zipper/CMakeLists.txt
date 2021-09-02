project(GenomeZipper LANGUAGES CXX)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE .)
target_sources(${PROJECT_NAME} INTERFACE genome_zipper.h)