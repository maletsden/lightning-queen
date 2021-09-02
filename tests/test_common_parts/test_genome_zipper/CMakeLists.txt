project(genome_zipper_test LANGUAGES CXX)

add_executable(${PROJECT_NAME} test_genome_zipper_zip.cpp)
target_link_libraries(${PROJECT_NAME} gtest gtest_main Threads::Threads GenomeZipper)
add_test(${PROJECT_NAME} ${PROJECT_NAME})