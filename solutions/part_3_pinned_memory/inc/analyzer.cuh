#ifndef PART_3_PINNED_MEMORY_ANALYZER_CUH
#define PART_3_PINNED_MEMORY_ANALYZER_CUH

#include <string>
#include <cuda.h>
#include <driver_types.h>

#include <PinnedFileHandler.cuh>
#include <thread_safe_queue/ThreadSafeQueue.h>

namespace analyzer {
  using RESULT_T = std::uint32_t;
  using QUEUE_T = ThreadSafeQueue<std::shared_ptr<PinnedFileHandler>>;

  constexpr RESULT_T CACHE_LINE_SIZE = 128;

  __global__ void analyze_genome(
      const char *device_genome_buffer, RESULT_T *results_vector, std::size_t genome_size
  );

  void analyze(QUEUE_T &genomes_queue);
}


#endif //PART_3_PINNED_MEMORY_ANALYZER_CUH