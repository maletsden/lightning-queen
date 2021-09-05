#include "gtest/gtest.h"

#include "test_genome_zipper_unzip.h"
#include "test_genome_zipper_zip.h"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}