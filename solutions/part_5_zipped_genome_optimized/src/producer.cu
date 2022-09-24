#include "producer.cuh"

#include <fstream>
#include <string>

#include <fs_handler/fs_handler.h>

void producer::produce(analyzer::QUEUE_T &genomes_queue, const std::string &genomes_directory,
                       const std::string &genomes_paths_file_path) {
  auto genomes_paths_file_handler = fs_handler::make_readable_file_handler(genomes_paths_file_path);

  for (std::string genome_path; std::getline(genomes_paths_file_handler.file, genome_path);) {
    if (genome_path.empty()) continue;

    const auto filepath = genomes_directory + genome_path;

    auto zipped_genome_file_handler = fs_handler::make_readable_binary_file_handler(filepath);
    std::string real_genome_size_str;
    fs_handler::getline(zipped_genome_file_handler, real_genome_size_str);
    const auto real_genome_size = std::stoi(real_genome_size_str);

    const auto filesize = fs_handler::get_filesize_till_end(zipped_genome_file_handler);

    analyzer::QUEUE_ITEM_T zipped_genome_handler{};
    zipped_genome_handler.real_size = real_genome_size;
    zipped_genome_handler.container = std::make_shared<PinnedMemoryHandler>(static_cast<size_t>(filesize),
                                                                            cudaHostAllocMapped);

    const auto buffer_ptr = fs_handler::read_file(zipped_genome_file_handler.file,
                                                  zipped_genome_handler.container->get_data(), filesize);

    if (buffer_ptr == nullptr) {
      std::cerr << "Failed reading file: " << filepath << std::endl;
      continue;
    }

    genomes_queue.enqueue(std::move(zipped_genome_handler));
  }

  // Add poison pill.
  genomes_queue.enqueue(analyzer::QUEUE_ITEM_T{});
}
