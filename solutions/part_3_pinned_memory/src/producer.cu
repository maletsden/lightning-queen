#include "producer.cuh"

#include <fstream>
#include <string>

#include <fs_handler/fs_handler.h>

void producer::produce(analyzer::QUEUE_T &genomes_queue, const std::string &genomes_directory,
                       const std::string &genomes_paths_file_path) {
  auto genomes_paths_file_handler = fs_handler::make_readable_file_handler(genomes_paths_file_path);

  std::string genome_path;
  while (std::getline(genomes_paths_file_handler.file, genome_path)) {
    if (genome_path.empty()) continue;

    const auto filepath = genomes_directory + genome_path;
    const auto filesize = fs_handler::get_filesize(filepath);

    const auto file_handler_ptr = std::make_shared<PinnedMemoryHandler>(static_cast<size_t>(filesize));

    const auto buffer_ptr = fs_handler::read_file(filepath, file_handler_ptr->get_data(), filesize);

    if (buffer_ptr == nullptr) {
      std::cerr << "Failed reading file: " << filepath << std::endl;
      continue;
    }

    genomes_queue.enqueue(file_handler_ptr);
  }

  // Add poison pill.
  genomes_queue.enqueue(std::make_shared<PinnedMemoryHandler>());
}
