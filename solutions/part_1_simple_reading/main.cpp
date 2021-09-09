#include <thread>
#include <iostream>

#include <analyzer.cuh>
#include <fs_handler/fs_handler.h>
#include <stopwatch/Stopwatch.h>

int main() {
  const Stopwatch stopwatch{"Total used time: "};

  const std::string genomes_directory = "../../data_generator/";
  const auto genomes_paths_file_path = genomes_directory + "genomes_paths.txt";

  ThreadSafeQueue<std::string> genomes_queue{};

  Stopwatch stopwatch_producer{""};

  std::thread producer{[&genomes_queue, &genomes_directory, &genomes_paths_file_path, &stopwatch_producer]() {
    stopwatch_producer = Stopwatch{"Total producer time: "};
    auto genomes_paths_file_handler = fs_handler::make_readable_file_handler(genomes_paths_file_path);

    std::string genome_path;
    while (std::getline(genomes_paths_file_handler.file, genome_path)) {
      if (genome_path.empty()) continue;

      auto file = fs_handler::read_file(genomes_directory + genome_path);

      if (file.empty()) {
        std::cerr << "Failed reading file: " << genome_path << std::endl;
        continue;
      }

      genomes_queue.enqueue(std::move(file));
    }

    // Add poison pill.
    genomes_queue.enqueue("");
    stopwatch_producer.stop();
  }};


  // Consumer.
  analyzer::analyze(genomes_queue);

  producer.join();

  return EXIT_SUCCESS;
}