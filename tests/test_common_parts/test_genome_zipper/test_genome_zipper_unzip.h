#include "gtest/gtest.h"

#include <genome_zipper/genome_zipper.h>

class TestGenomeZipperUnzip : public ::testing::Test {
protected:
  static void CompareResults(const std::string &genome, const char *expected_genome, size_t expected_size) {
    EXPECT_EQ(genome.size(), expected_size);
    EXPECT_STREQ(genome.data(), expected_genome);
  }
};

TEST_F(TestGenomeZipperUnzip, SimpleZeroNTest) {
  constexpr auto expected_genome_size = 3;
  constexpr char expected_genome[] = "ACG";
  constexpr char zipped_genome[] = {0b00000110};
  const auto genome = genome_zipper::unzip(zipped_genome, expected_genome_size);
  CompareResults(genome, expected_genome, expected_genome_size);
}

TEST_F(TestGenomeZipperUnzip, SimpleZippedGenomeTest) {
  constexpr auto expected_genome_size = 3;
  constexpr char expected_genome[] = "ACG";

  const genome_zipper::ZippedGenome<std::string> zipped_genome{expected_genome_size, std::string(1, 0b00000110)};
  const auto genome = genome_zipper::unzip(zipped_genome);
  CompareResults(genome, expected_genome, expected_genome_size);
}

TEST_F(TestGenomeZipperUnzip, SimpleOneNTest) {
  constexpr auto expected_genome_size = 3;
  constexpr char expected_genome[] = "NAC";
  constexpr char zipped_genome[] = {0b01000001};
  const auto genome = genome_zipper::unzip(zipped_genome, expected_genome_size);
  CompareResults(genome, expected_genome, expected_genome_size);

  // ---------- //

  constexpr char expected_genome2[] = "ANC";
  constexpr char zipped_genome2[] = {0b01010001};
  const auto genome2 = genome_zipper::unzip(zipped_genome2, expected_genome_size);
  CompareResults(genome2, expected_genome2, expected_genome_size);

  // ---------- //

  constexpr char expected_genome3[] = "ACN";
  constexpr char zipped_genome3[] = {0b01100001};
  const auto genome3 = genome_zipper::unzip(zipped_genome3, expected_genome_size);
  CompareResults(genome3, expected_genome3, expected_genome_size);
}

TEST_F(TestGenomeZipperUnzip, SimpleTwoNTest) {
  constexpr auto expected_genome_size = 3;
  constexpr char expected_genome[] = "NNA";
  constexpr char zipped_genome[] = {static_cast<char>(0b10100000)};
  const auto genome = genome_zipper::unzip(zipped_genome, expected_genome_size);
  CompareResults(genome, expected_genome, expected_genome_size);

  // ---------- //

  constexpr char expected_genome2[] = "NAN";
  constexpr char zipped_genome2[] = {static_cast<char>(0b10010000)};
  const auto genome2 = genome_zipper::unzip(zipped_genome2, expected_genome_size);
  CompareResults(genome2, expected_genome2, expected_genome_size);

  // ---------- //

  constexpr char expected_genome3[] = "ANN";
  constexpr char zipped_genome3[] = {static_cast<char>(0b10000000)};
  const auto genome3 = genome_zipper::unzip(zipped_genome3, expected_genome_size);
  CompareResults(genome3, expected_genome3, expected_genome_size);
}

TEST_F(TestGenomeZipperUnzip, SimpleThreeNTest) {
  constexpr auto expected_genome_size = 3;
  constexpr char expected_genome[] = "NNN";
  constexpr char zipped_genome[] = {static_cast<char>(0b11000000)};
  const auto genome = genome_zipper::unzip(zipped_genome, expected_genome_size);
  CompareResults(genome, expected_genome, expected_genome_size);
}

TEST_F(TestGenomeZipperUnzip, LongGenomeTest) {
  constexpr auto expected_genome_size = 9;
  constexpr char expected_genome[] = "ACGANCANN";
  constexpr char zipped_genome[] = {0b00000110, 0b01010001, static_cast<char>(0b10000000)};
  const auto genome = genome_zipper::unzip(zipped_genome, expected_genome_size);
  CompareResults(genome, expected_genome, expected_genome_size);
}

TEST_F(TestGenomeZipperUnzip, GenomeSize4Test) {
  constexpr auto expected_genome_size = 4;
  constexpr char expected_genome[] = "ACGA";
  constexpr char zipped_genome[] = {0b00000110, 0b00000000};
  const auto genome = genome_zipper::unzip(zipped_genome, expected_genome_size);
  CompareResults(genome, expected_genome, expected_genome_size);

  // ---------- //

  constexpr char expected_genome2[] = "ACGN";
  constexpr char zipped_genome2[] = {0b00000110, 0b01000000};
  const auto genome2 = genome_zipper::unzip(zipped_genome2, expected_genome_size);
  CompareResults(genome2, expected_genome2, expected_genome_size);
}

TEST_F(TestGenomeZipperUnzip, GenomeSize5Test) {
  constexpr auto expected_genome_size = 5;
  constexpr char expected_genome[] = "ACGAA";
  constexpr char zipped_genome[] = {0b00000110, 0b00000000};
  const auto genome = genome_zipper::unzip(zipped_genome, expected_genome_size);
  CompareResults(genome, expected_genome, expected_genome_size);

  // ---------- //

  constexpr char expected_genome2[] = "ACGNA";
  constexpr char zipped_genome2[] = {0b00000110, 0b01000000};
  const auto genome2 = genome_zipper::unzip(zipped_genome2, expected_genome_size);
  CompareResults(genome2, expected_genome2, expected_genome_size);

  // ---------- //

  constexpr char expected_genome3[] = "ACGNA";
  constexpr char zipped_genome3[] = {0b00000110, 0b01000000};
  const auto genome3 = genome_zipper::unzip(zipped_genome3, expected_genome_size);
  CompareResults(genome3, expected_genome3, expected_genome_size);

  // ---------- //

  constexpr char expected_genome4[] = "ACGNN";
  constexpr char zipped_genome4[] = {0b00000110, static_cast<char>(0b10100000)};
  const auto genome4 = genome_zipper::unzip(zipped_genome4, expected_genome_size);
  CompareResults(genome4, expected_genome4, expected_genome_size);
}