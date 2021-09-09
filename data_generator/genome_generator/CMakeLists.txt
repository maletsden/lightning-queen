project(genome_generator LANGUAGES CXX)

include_directories(inc)

add_executable(${PROJECT_NAME} main.cpp src/genome_generator.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads FsHandler stdc++fs)
