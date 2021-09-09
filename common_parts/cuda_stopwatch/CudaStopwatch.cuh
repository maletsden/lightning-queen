#ifndef GENOME_ANALYZER_CUDA_STOPWATCH_H
#define GENOME_ANALYZER_CUDA_STOPWATCH_H

#include <cuda_runtime.h>
#include <cuda_validator/cuda_validator.h>

class CudaStopwatch {
public:
  CudaStopwatch();
  ~CudaStopwatch();

  void start(size_t memory_size);
  void stop();

private:
  cudaEvent_t m_startEvent{}, m_stopEvent{};
  float m_total_time = 0.f;
  size_t m_memory_size = 0;
};

#endif //GENOME_ANALYZER_CUDA_STOPWATCH_H
