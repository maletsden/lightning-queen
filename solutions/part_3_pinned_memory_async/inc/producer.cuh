#ifndef PART_3_PINNED_MEMORY_ASYNC_PRODUCER_CUH
#define PART_3_PINNED_MEMORY_ASYNC_PRODUCER_CUH

#include "analyzer.cuh"

namespace producer {
  void produce(analyzer::QUEUE_T &genomes_queue, const std::string &genomes_directory,
               const std::string &genomes_paths_file_path);
}

#endif //PART_3_PINNED_MEMORY_ASYNC_PRODUCER_CUH
