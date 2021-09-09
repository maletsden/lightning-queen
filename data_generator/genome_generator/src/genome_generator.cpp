#include "genome_generator.h"

#include <array>
#include <cstring>
#include <random>
#include <algorithm>
#include <iostream>

#include <fs_handler/fs_handler.h>

void genome_generator::generate(const std::vector<std::string> &genomes_paths, size_t genome_size) {
  // generated genome will consist equal amount of all nucleotides (20% = 20 MB)
  const size_t num_same_nucleotides = genome_size / 5;
  constexpr std::array<char, 4> nucleotides = {'A', 'C', 'G', 'T'};
  constexpr char default_nucleotide = 'N';

  // generate genome
  std::string genome(genome_size, default_nucleotide);
  auto start = const_cast<char *>(genome.data());

  for (char nucleotide : nucleotides) {
    memset(start, nucleotide, num_same_nucleotides);
    start += num_same_nucleotides;
  }

  for (const auto &genome_path: genomes_paths) {
    // shuffle genome
    std::shuffle(genome.begin(), genome.end(), std::mt19937(std::random_device()()));

    // save to file
    fs_handler::write_file("../" + genome_path, genome);

    std::cout << "Genome successfully written to: " << genome_path << std::endl;
  }
}
