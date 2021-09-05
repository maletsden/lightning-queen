project(test_genome_zipper LANGUAGES CXX)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} gtest gtest_main Threads::Threads GenomeZipper)
add_test(${PROJECT_NAME} ${PROJECT_NAME})