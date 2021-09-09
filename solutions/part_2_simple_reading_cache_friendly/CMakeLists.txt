project(part_2_simple_reading_cache_friendly LANGUAGES	CUDA CXX)

include_directories(inc)

add_library(${PROJECT_NAME}_Analyzer STATIC src/analyzer.cu inc/analyzer.cuh)
set_target_properties(${PROJECT_NAME}_Analyzer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(${PROJECT_NAME}_Analyzer PUBLIC "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
target_link_libraries(${PROJECT_NAME}_Analyzer PRIVATE CudaStopwatch)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE
        ${PROJECT_NAME}_Analyzer
        CudaValidator
        Validator
        ThreadSafeQueue
        FsHandler
        Threads::Threads
        Stopwatch)