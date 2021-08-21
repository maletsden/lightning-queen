#include <random>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <array>
#include <fstream>
#include <string>
#include <iostream>
#include <thread>
#include <filesystem>
#include <cstring>
#include <cassert>

void generate_genomes(const std::vector<std::string> &genomes_paths);

int main(int argc, char **argv) {
  // Check the number of parameters
  if (argc < 3) {
    // Tell the user how to run the program
    std::cerr << "Usage: ./data_generator <GENOMES_NUM> <THREADS_NUM>" << std::endl
              << "<GENOMES_NUM> - number of genomes to generate" << std::endl
              << "<THREADS_NUM> - number of threads to use for generation" << std::endl;

    return 1;
  }

  const auto genomes_num = std::stoi(argv[1]);
  const auto thread_num = std::stoi(argv[2]);

  std::vector<std::string> genomes_paths;
  genomes_paths.reserve(genomes_num);

  for (auto i = 0; i < genomes_num; ++i) {
    genomes_paths.emplace_back("./bank/genome_" + std::to_string(i) + ".fasta");
  }

  // start generating genomes
  std::vector<std::thread> threads;
  threads.reserve(thread_num);

  auto genomes_paths_start = genomes_paths.cbegin();
  const auto genomes_per_thread = genomes_num / thread_num;

  std::cout << "Start generating genomes..." << std::endl;
  for (auto i = 0; i < thread_num; ++i) {
    std::vector<std::string> thread_genomes_paths{
        genomes_paths_start, std::min(genomes_paths_start + genomes_per_thread, genomes_paths.cend())
    };
    threads.emplace_back(generate_genomes, std::move(thread_genomes_paths));
    genomes_paths_start += genomes_per_thread;
  }

  for (auto &thread: threads) {
    thread.join();
  }

  // save genomes paths
  std::ofstream genomes_paths_file;
  genomes_paths_file.open("genomes_paths.txt");
  for (const auto &genome_path: genomes_paths) {
    genomes_paths_file << genome_path << std::endl;
  }
  genomes_paths_file.close();

  std::cout << "Genomes successfully generated." << std::endl;

  return 0;
}


void generate_genomes(const std::vector<std::string> &genomes_paths) {
  constexpr std::uint32_t genome_size = 100 * 1024 * 1024; // 100 MB
  // generated genome will consist equal amount of all nucleotides (20% = 20 MB)
  constexpr std::uint32_t num_same_nucleotides = 20 * 1024 * 1024;
  constexpr std::array<char, 4> nucleotides{'A', 'C', 'G', 'T'};
  constexpr char default_nucleotide = 'N';

  // generate genome
  std::string genome(genome_size, default_nucleotide);
  auto start = const_cast<char *>(genome.data());

  for (char nucleotide : nucleotides) {
    memset(start, nucleotide, num_same_nucleotides);
    start += num_same_nucleotides;
  }

  for (char nucleotide : nucleotides) {
    assert(std::count(genome.cbegin(), genome.cend(), nucleotide) == num_same_nucleotides);
  }
  assert(std::count(genome.cbegin(), genome.cend(), default_nucleotide) == num_same_nucleotides);


  for (const auto &genome_path: genomes_paths) {
    // shuffle genome
    std::shuffle(genome.begin(), genome.end(), std::mt19937(std::random_device()()));

    // save to file
    std::ofstream genome_file;
    genome_file.open(genome_path);
    genome_file << genome;
    genome_file.close();

    std::cout << "Genome successfully written to: " << genome_path << std::endl;
  }
}