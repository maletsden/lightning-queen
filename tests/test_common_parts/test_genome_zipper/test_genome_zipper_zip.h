#include "gtest/gtest.h"

#include <genome_zipper/genome_zipper.h>

class TestGenomeZipperZip : public ::testing::Test {
protected:
  static void CompareResults(const genome_zipper::ZippedGenome<std::string> &zipped,
                             const genome_zipper::ZippedGenome<std::string> &expected) {
    EXPECT_EQ(zipped.container.size(), expected.container.size());
    EXPECT_EQ(zipped.real_size, expected.real_size);
    EXPECT_STREQ(zipped.container.data(), expected.container.data());
  }
};

TEST_F(TestGenomeZipperZip, SimpleZeroNTestArrayInput) {
  constexpr auto genome_size = 3;
  constexpr char genome[] = "ACG";

  genome_zipper::ZippedGenome<std::string> expected_result;
  expected_result.real_size = genome_size;
  expected_result.container.push_back(0b00000110);

  const auto zipped_genome = genome_zipper::zip(genome);

  CompareResults(zipped_genome, expected_result);
}

TEST_F(TestGenomeZipperZip, SimpleZeroNTestPointerInput) {
  constexpr auto genome_size = 3;
  constexpr char genome[] = "ACG";

  genome_zipper::ZippedGenome<std::string> expected_result;
  expected_result.real_size = genome_size;
  expected_result.container.push_back(0b00000110);

  const auto zipped_genome = genome_zipper::zip(genome, genome_size);

  CompareResults(zipped_genome, expected_result);
}

TEST_F(TestGenomeZipperZip, SimpleZeroNTestStringInput) {
  const std::string genome = "ACG";

  genome_zipper::ZippedGenome<std::string> expected_result;
  expected_result.real_size = genome.size();
  expected_result.container.push_back(0b00000110);

  const auto zipped_genome = genome_zipper::zip(genome);

  CompareResults(zipped_genome, expected_result);
}

TEST_F(TestGenomeZipperZip, SimpleOneNTest) {
  std::string genome = "NAA";

  genome_zipper::ZippedGenome<std::string> expected_result;
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

  genome_zipper::ZippedGenome<std::string> expected_result;
  expected_result.real_size = genome.size();
  expected_result.container.push_back(static_cast<char>(0b10100100));

  auto zipped_genome = genome_zipper::zip(genome);

  CompareResults(zipped_genome, expected_result);

  // ---------- //

  genome = "NAN";

  expected_result.real_size = genome.size();
  expected_result.container[0] = static_cast<char>(0b10010000);

  zipped_genome = genome_zipper::zip(genome);

  CompareResults(zipped_genome, expected_result);

  // ---------- //

  genome = "ANN";

  expected_result.real_size = genome.size();
  expected_result.container[0] = static_cast<char>(0b10000000);

  zipped_genome = genome_zipper::zip(genome);

  CompareResults(zipped_genome, expected_result);
}

TEST_F(TestGenomeZipperZip, SimpleThreeNTest) {
  const std::string genome = "NNN";

  genome_zipper::ZippedGenome<std::string> expected_result;
  expected_result.real_size = genome.size();
  expected_result.container.push_back(static_cast<char>(0b11000000));

  const auto zipped_genome = genome_zipper::zip(genome);

  CompareResults(zipped_genome, expected_result);
}

TEST_F(TestGenomeZipperZip, LongGenomeTest) {
  const std::string genome = "ACGTNA";

  genome_zipper::ZippedGenome<std::string> expected_result;
  expected_result.real_size = genome.size();
  expected_result.container.push_back(0b00000110);
  expected_result.container.push_back(0b01011100);

  const auto zipped_genome = genome_zipper::zip(genome);

  CompareResults(zipped_genome, expected_result);
}

TEST_F(TestGenomeZipperZip, NotDiv3GenomeTest) {
  // expected to interpret as "ACGTAA" (zeros in the end)
  std::string genome = "ACGT";

  genome_zipper::ZippedGenome<std::string> expected_result;
  expected_result.real_size = genome.size();
  expected_result.container.push_back(0b00000110);
  expected_result.container.push_back(0b00110000);

  auto zipped_genome = genome_zipper::zip(genome);

  CompareResults(zipped_genome, expected_result);

  // ---------- //
  // expected to interpret as "ACGNAA" (zeros in the end)
  genome = "ACGN";

  expected_result.real_size = genome.size();
  expected_result.container[1] = 0b01000000;

  zipped_genome = genome_zipper::zip(genome);

  CompareResults(zipped_genome, expected_result);

  // ---------- //
  // expected to interpret as "ACGTAA" (zeros in the end)
  genome = "ACGTA";

  expected_result.real_size = genome.size();
  expected_result.container[1] = 0b00110000;

  zipped_genome = genome_zipper::zip(genome);

  CompareResults(zipped_genome, expected_result);

  // ---------- //
  // expected to interpret as "ACGTNA" (zeros in the end)
  genome = "ACGTN";

  expected_result.real_size = genome.size();
  expected_result.container[1] = 0b01011100;

  zipped_genome = genome_zipper::zip(genome);

  CompareResults(zipped_genome, expected_result);

  // ---------- //
  // expected to interpret as "ACGNNA" (zeros in the end)
  genome = "ACGNN";

  expected_result.real_size = genome.size();
  expected_result.container[1] = static_cast<char>(0b10100000);

  zipped_genome = genome_zipper::zip(genome);

  CompareResults(zipped_genome, expected_result);
}
