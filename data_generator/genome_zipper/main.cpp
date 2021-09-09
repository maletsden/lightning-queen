#include <iostream>
#include <vector>
#include <thread>
#include <filesystem>
#include <string>

#include <genome_zipper/genome_zipper.h>
#include <fs_handler/fs_handler.h>
#include <thread_safe_queue/ThreadSafeQueue.h>

int main(int argc, char **argv) {
  // Check the number of parameters
  if (argc < 2) {
    // Tell the user how to run the program
    std::cerr << "Usage: ./test_genome_zipper <THREADS_NUM>" << std::endl
              << "<THREADS_NUM> - number of threads to use for zipping" << std::endl;

    return 1;
  }

  const auto thread_num = std::max(std::stoi(argv[1]), 1);
  // reserve 2 threads for file reader and writer
  const auto zipper_threads_num = std::max(thread_num - 2, 1);

  constexpr auto genome_paths_file_path = "../genomes_paths.txt";

  // read genomes paths
  std::ifstream genomes_paths_file(genome_paths_file_path);

  std::string genome_path;
  std::vector<std::string> genomes_paths;
  while (std::getline(genomes_paths_file, genome_path)) {
    genomes_paths.push_back("../" + genome_path);
  }

  ThreadSafeQueue<std::pair<size_t, std::string>> genomes_queue{};
  ThreadSafeQueue<std::pair<size_t, genome_zipper::ZippedGenome>> zipped_genomes_queue{};

  std::thread reader{[&genomes_queue, &genomes_paths, zipper_threads_num]() {
    size_t genome_idx = 0;
    for (const auto &genome_path: genomes_paths) {
      auto genome_data = fs_handler::read_file(genome_path);

      if (genome_data.empty()) {
        std::cerr << "Failed reading file: " << genome_path << std::endl;
        continue;
      }

      genomes_queue.enqueue(std::make_pair(genome_idx++, std::move(genome_data)));
    }

    // Add poison pills.
    for (auto i = 0; i < zipper_threads_num; ++i) {
      genomes_queue.enqueue(std::make_pair(-1, ""));
    }
  }};


  std::vector<std::thread> zipper_threads;
  zipper_threads.reserve(zipper_threads_num);

  const std::string output_directory = "../zipped_bank";
  std::filesystem::create_directories(output_directory);

  std::cout << "Start zipping genomes..." << std::endl;

  const auto zipper = [&genomes_queue, &zipped_genomes_queue]() {
    while (true) {
      const auto genome_data_pair = genomes_queue.dequeue();
      const auto genome_idx = std::get<0>(genome_data_pair);
      const auto genome_data = std::get<1>(genome_data_pair);

      // Catch poison pill.
      if (genome_idx == -1 && genome_data.empty()) break;

      auto zipped_genome_data = genome_zipper::zip(genome_data);

      zipped_genomes_queue.enqueue(std::make_pair(genome_idx, std::move(zipped_genome_data)));
    }

    // Add poison pill.
    zipped_genomes_queue.enqueue(std::make_pair(-1, genome_zipper::ZippedGenome{0, ""}));
  };

  for (auto i = 0; i < zipper_threads_num; ++i) {
    zipper_threads.emplace_back(zipper);
  }

  // writer
  {
    size_t poisson_pills_num = 0;

    while (true) {
      const auto zipped_genome_data_pair = zipped_genomes_queue.dequeue();
      const auto zipped_genome_idx = std::get<0>(zipped_genome_data_pair);
      const auto zipped_genome_data = std::get<1>(zipped_genome_data_pair);

      // Catch poison pill.
      if (zipped_genome_data.is_empty()) {
        if (++poisson_pills_num == zipper_threads_num) {
          break;
        }
        continue;
      }

      const auto zipped_genome_path =
          output_directory + "/zipped_genome_" + std::to_string(zipped_genome_idx) + ".3zip";
      auto file_handler = fs_handler::make_writable_file_handler(zipped_genome_path);

      file_handler << std::to_string(zipped_genome_data.real_size) << zipped_genome_data.container;

      std::cout << "Successfully zipped genome " << zipped_genome_path << std::endl;
    }
  }

  reader.join();
  for (auto &zipper_thread: zipper_threads) {
    zipper_thread.join();
  }

}