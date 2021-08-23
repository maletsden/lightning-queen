#include <fstream>
#include <thread>
#include <iostream>

#include <analyzer.cuh>
#include <producer.cuh>
#include <memory>

int main() {
  const std::string genomes_directory = "../../data_generator/";
  const auto genomes_paths_file_path = genomes_directory + "genomes_paths.txt";

  analyzer::QUEUE_T genomes_queue{};

  std::thread producer{[&genomes_queue, &genomes_directory, &genomes_paths_file_path]() {
    producer::produce(genomes_queue, genomes_directory, genomes_paths_file_path);
  }};

  // Consumer.
  analyzer::analyze(genomes_queue);

  producer.join();

  return EXIT_SUCCESS;
}