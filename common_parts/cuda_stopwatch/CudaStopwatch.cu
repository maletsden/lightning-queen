#include "CudaStopwatch.cuh"

#include <iostream>

CudaStopwatch::CudaStopwatch() {
  cuda_validator::check_error(cudaEventCreate(&m_startEvent));
  cuda_validator::check_error(cudaEventCreate(&m_stopEvent));
}

CudaStopwatch::~CudaStopwatch() {
  std::cout << "Total analyzer time: " << m_total_time * 1e3 << " μs." << std::endl;

  cuda_validator::check_error(cudaEventDestroy(m_startEvent));
  cuda_validator::check_error(cudaEventDestroy(m_stopEvent));
}

void CudaStopwatch::start(size_t memory_size) {
  m_memory_size = memory_size;
  cuda_validator::check_error(cudaEventRecord(m_startEvent, nullptr));
}

void CudaStopwatch::stop() {
  cuda_validator::check_error(cudaEventRecord(m_stopEvent, nullptr));
  cuda_validator::check_error(cudaEventSynchronize(m_stopEvent));

  float time;
  cuda_validator::check_error(cudaEventElapsedTime(&time, m_startEvent, m_stopEvent));

  m_total_time += time;

  std::cout << "Transfer size (MB): " << m_memory_size / (1024 * 1024) << std::endl
            << "  Host to Device bandwidth (GB/s): " << static_cast<double>(m_memory_size) * 1e-6 / time
            << std::endl;
}
