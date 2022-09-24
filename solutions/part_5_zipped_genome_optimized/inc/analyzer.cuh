#ifndef PART_5_ZIPPED_GENOME_OPTIMIZED_ANALYZER_CUH
#define PART_5_ZIPPED_GENOME_OPTIMIZED_ANALYZER_CUH

#include <cuda_runtime.h>
#include <driver_types.h>
#include <memory>

#include <pinned_memory_handler/PinnedMemoryHandler.cuh>
#include <thread_safe_queue/ThreadSafeQueue.h>
#include <genome_zipper/ZippedGenome.h>

namespace analyzer {
  using RESULT_T = std::uint32_t;
  using QUEUE_ITEM_T = genome_zipper::ZippedGenome<std::shared_ptr<PinnedMemoryHandler>>;
  using QUEUE_T = ThreadSafeQueue<QUEUE_ITEM_T>;

  constexpr RESULT_T CACHE_LINE_SIZE = 128;
  constexpr RESULT_T EXPECTED_GENOME_SIZE = (100 * 1024 * 1024 + 3 - 1) / 3; // ~33.33 MB

  __global__ void analyze_genome(
      const char *device_genome_buffer, RESULT_T *results_vector, std::size_t zipped_genome_size,
      std::size_t real_genome_size
  );

  void analyze(QUEUE_T &genomes_queue);
}


#endif //PART_5_ZIPPED_GENOME_OPTIMIZED_ANALYZER_CUH