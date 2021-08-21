#ifndef PART_1_SIMPLE_READING_ANALYZER_CUH
#define PART_1_SIMPLE_READING_ANALYZER_CUH

#include <string>
#include <cuda.h>
#include <driver_types.h>

#include <thread_safe_queue/ThreadSafeQueue.h>

namespace analyzer {
  using RESULT_T = std::uint32_t;

  __global__ void analyzeGenome(
      const char *device_genome_buffer, RESULT_T *results_vector, std::size_t genome_size
  );

  void analyze(ThreadSafeQueue<std::string> &genomes_queue);
}
#endif //PART_1_SIMPLE_READING_ANALYZER_CUH