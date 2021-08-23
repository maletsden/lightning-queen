project(PinnedMemoryHandler LANGUAGES CUDA CXX)

include_directories(..)

add_library(${PROJECT_NAME} STATIC
        PinnedMemoryHandler.cuh
        PinnedMemoryHandler.cu)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_link_libraries(${PROJECT_NAME} PRIVATE CudaValidator)