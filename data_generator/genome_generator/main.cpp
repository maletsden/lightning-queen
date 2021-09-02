#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <iostream>
#include <thread>

#include <genome_generator.h>

int main(int argc, char **argv) {
  // Check the number of parameters
  if (argc < 3) {
    // Tell the user how to run the program
    std::cerr << "Usage: ./data_generator <GENOMES_NUM> <THREADS_NUM>" << std::endl
              << "<GENOMES_NUM> - number of genomes to generate" << std::endl
              << "<THREADS_NUM> - number of threads to use for generation" << std::endl;

    return 1;
  }

  const auto genomes_num = std::max(std::stoi(argv[1]), 0);
  const auto thread_num = std::max(std::stoi(argv[2]), 1);

  std::vector<std::string> genomes_paths;
  genomes_paths.reserve(genomes_num);

  for (auto i = 0; i < genomes_num; ++i) {
    genomes_paths.emplace_back("../bank/genome_" + std::to_string(i) + ".fasta");
  }

  // start generating genomes
  std::vector<std::thread> threads;
  threads.reserve(thread_num);

  auto genomes_paths_start = genomes_paths.cbegin();
  const auto genomes_per_thread = (genomes_num + thread_num - 1) / thread_num;

  constexpr size_t genome_size = 100 * 1024 * 1024; // 100 MB

  std::cout << "Start generating genomes..." << std::endl;
  for (auto i = 0; i < thread_num; ++i) {
    std::vector<std::string> thread_genomes_paths{
        genomes_paths_start, std::min(genomes_paths_start + genomes_per_thread, genomes_paths.cend())
    };
    threads.emplace_back(genome_generator::generate, std::move(thread_genomes_paths), genome_size);
    genomes_paths_start += genomes_per_thread;
  }

  for (auto &thread: threads) {
    thread.join();
  }

  // save genomes paths
  std::ofstream genomes_paths_file;
  genomes_paths_file.open("../genomes_paths.txt");
  for (const auto &genome_path: genomes_paths) {
    genomes_paths_file << genome_path << std::endl;
  }
  genomes_paths_file.close();

  std::cout << "Genomes successfully generated." << std::endl;

  return 0;
}