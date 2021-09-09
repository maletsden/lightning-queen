#include <thread>

#include <analyzer.cuh>
#include <producer.cuh>
#include <stopwatch/Stopwatch.h>

int main() {
  const Stopwatch stopwatch{"Total used time: "};
  const std::string genomes_directory = "../../data_generator/";
  const auto genomes_paths_file_path = genomes_directory + "genomes_paths.txt";

  analyzer::QUEUE_T genomes_queue{};

  Stopwatch stopwatch_producer{""};

  std::thread producer{[&genomes_queue, &genomes_directory, &genomes_paths_file_path, &stopwatch_producer]() {
    stopwatch_producer = Stopwatch{"Total producer time: "};
    producer::produce(genomes_queue, genomes_directory, genomes_paths_file_path);
    stopwatch_producer.stop();
  }};

  // Consumer.
  analyzer::analyze(genomes_queue);

  producer.join();

  return EXIT_SUCCESS;
}