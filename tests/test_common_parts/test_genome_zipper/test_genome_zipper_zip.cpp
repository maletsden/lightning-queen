#include "gtest/gtest.h"

#include <genome_zipper/genome_zipper.h>
#include <bitset>

// The fixture for testing class Foo.
class TestGenomeZipperZip : public ::testing::Test {
protected:
  // You can remove any or all of the following functions if their bodies would
  // be empty.

  TestGenomeZipperZip() {
    // You can do set-up work for each test here.
  }

  ~TestGenomeZipperZip() override {
    // You can do clean-up work that doesn't throw exceptions here.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  void SetUp() override {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  void TearDown() override {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }

  void CompareResults(const genome_zipper::ZippedGenome &zipped, const genome_zipper::ZippedGenome &expected) {
    EXPECT_EQ(zipped.container.size(), expected.container.size());
    EXPECT_EQ(zipped.real_size, expected.real_size);
    EXPECT_STREQ(zipped.container.data(), expected.container.data());
  }

  // Class members declared here can be used by all tests in the test suite
  // for Foo.
};

TEST_F(TestGenomeZipperZip, SimpleZeroNTestArrayInput) {
  constexpr auto genome_size = 3;
  constexpr char genome[] = "ACG";

  genome_zipper::ZippedGenome expected_result;
  expected_result.real_size = genome_size;
  expected_result.container.push_back(0b00000110);

  const auto zipped_genome = genome_zipper::zip(genome);

  CompareResults(zipped_genome, expected_result);
}

TEST_F(TestGenomeZipperZip, SimpleZeroNTestPointerInput) {
  constexpr auto genome_size = 3;
  constexpr char genome[] = "ACG";

  genome_zipper::ZippedGenome expected_result;
  expected_result.real_size = genome_size;
  expected_result.container.push_back(0b00000110);

  const auto zipped_genome = genome_zipper::zip(genome, genome_size);

  CompareResults(zipped_genome, expected_result);
}

TEST_F(TestGenomeZipperZip, SimpleZeroNTestStringInput) {
  const std::string genome = "ACG";

  genome_zipper::ZippedGenome expected_result;
  expected_result.real_size = genome.size();
  expected_result.container.push_back(0b00000110);

  const auto zipped_genome = genome_zipper::zip(genome);

  CompareResults(zipped_genome, expected_result);
}

TEST_F(TestGenomeZipperZip, SimpleOneNTest) {
  std::string genome = "NAA";

  genome_zipper::ZippedGenome expected_result;
  expected_result.real_size = genome.size();
  expected_result.container.push_back(0b01000000);

  auto zipped_genome = genome_zipper::zip(genome);

  CompareResults(zipped_genome, expected_result);

  // ---------- //

  genome = "ANA";

  expected_result.real_size = genome.size();
  expected_result.container[0] = 0b01010000;

  zipped_genome = genome_zipper::zip(genome);

  CompareResults(zipped_genome, expected_result);

  // ---------- //

  genome = "AAN";

  expected_result.real_size = genome.size();
  expected_result.container[0] = 0b01100000;

  zipped_genome = genome_zipper::zip(genome);

  CompareResults(zipped_genome, expected_result);
}

TEST_F(TestGenomeZipperZip, SimpleTwoNTest) {
  std::string genome = "NNC";

  genome_zipper::ZippedGenome expected_result;
  expected_result.real_size = genome.size();
  expected_result.container.push_back(0b10100100);

  auto zipped_genome = genome_zipper::zip(genome);

  CompareResults(zipped_genome, expected_result);

  // ---------- //

  genome = "NAN";

  expected_result.real_size = genome.size();
  expected_result.container[0] = 0b10010000;

  zipped_genome = genome_zipper::zip(genome);

  CompareResults(zipped_genome, expected_result);

  // ---------- //

  genome = "ANN";

  expected_result.real_size = genome.size();
  expected_result.container[0] = 0b10000000;

  zipped_genome = genome_zipper::zip(genome);

  CompareResults(zipped_genome, expected_result);
}

TEST_F(TestGenomeZipperZip, SimpleThreeNTest) {
  const std::string genome = "NNN";

  genome_zipper::ZippedGenome expected_result;
  expected_result.real_size = genome.size();
  expected_result.container.push_back(0b11000000);

  const auto zipped_genome = genome_zipper::zip(genome);

  CompareResults(zipped_genome, expected_result);
}



int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}