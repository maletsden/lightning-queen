#include "inc/analyzer.cuh"

std::string read_file(const std::string &filepath) {

  std::ifstream file(filepath, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::string buffer(size, '0');
  if (file.read(const_cast<char *>(buffer.data()), size)) {
    return buffer;
  }

  return "";
}

int main() {
  constexpr auto genomes_directory = "../data/";
  constexpr auto genomes_paths_file_path = "../data/genomes_paths.txt";

  ThreadSafeQueue<std::string> genomes_queue{};

  std::thread producer{[&genomes_queue]() {
    std::ifstream genomes_paths_file(genomes_paths_file_path);

    std::string genome_path;
    while (std::getline(genomes_paths_file, genome_path)) {

      auto relative_path = genomes_directory + genome_path;
      auto file = read_file(relative_path);

      if (file.empty()) {
        std::cerr << "Failed reading file: " << genome_path << std::endl;
        continue;
      }

      genomes_queue.enqueue(file);
    }

    // Add poison pill.
    genomes_queue.enqueue("");
  }};



  // Consumer.
  run_analyzer(genomes_queue);

  producer.join();

  return 0;
}