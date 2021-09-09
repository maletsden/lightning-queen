#include "analyzer.cuh"

#include <iostream>
#include <vector>
#include <array>
#include <cassert>

#include <validator/validator.h>
#include <cuda_validator/cuda_validator.h>
#include <cuda_stopwatch/CudaStopwatch.cuh>


__global__ void analyzer::analyze_genome(
    const char *device_genome_buffer, RESULT_T *results_vector, std::size_t genome_size
) {
  const auto i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= genome_size) return;

  const std::uint8_t nucleotide = device_genome_buffer[i];

  constexpr std::uint8_t charACode = 'A';

  atomicAdd(results_vector + threadIdx.x * CACHE_LINE_SIZE + nucleotide - charACode, 1);
}


void analyzer::analyze(QUEUE_T &genomes_queue) {

  constexpr RESULT_T expected_genome_size = 100 * 1024 * 1024; // 100 MB
  constexpr int threadsPerBlock = 256;
  constexpr int blocksPerGrid = (expected_genome_size + threadsPerBlock - 1) / threadsPerBlock;
  cudaDeviceProp prop{};
  cuda_validator::check_error(cudaGetDeviceProperties(&prop, 0));

  std::cout << "Device: " << prop.name << std::endl;

  if (!prop.canMapHostMemory) {
    std::cerr << "Mapped Host memory is not supported for this device";
    exit(EXIT_FAILURE);
  }
  cudaSetDeviceFlags(cudaDeviceMapHost);

  // Allocate result vector on device.
  RESULT_T *results_vector;
  constexpr RESULT_T results_buffer_size = CACHE_LINE_SIZE * threadsPerBlock * 20;
  constexpr RESULT_T results_buffer_size_in_bytes = results_buffer_size * sizeof(RESULT_T);

  cuda_validator::check_error(cudaMalloc(&results_vector, results_buffer_size_in_bytes)); // device

  std::vector<RESULT_T> host_result_vector(results_buffer_size);
  std::vector<RESULT_T> host_result_vector_total(results_buffer_size / threadsPerBlock);

  CudaStopwatch cuda_stopwatch;


  while (true) {
    auto file_handler = genomes_queue.dequeue();

    // Catch poison pill.
    if (file_handler->is_empty()) break;

    const auto filesize = file_handler->get_size();

    cuda_stopwatch.start(filesize);

    cuda_validator::check_error(cudaMemset(results_vector, 0, results_buffer_size_in_bytes));

    char *device_genome_buffer;
    cuda_validator::check_error(cudaHostGetDevicePointer(&device_genome_buffer, file_handler->get_data(), 0));

    // Invoke kernel.
    analyze_genome<<<blocksPerGrid, threadsPerBlock>>>(device_genome_buffer, results_vector, filesize);

    cuda_validator::check_error(
        cudaMemcpy(host_result_vector.data(), results_vector, results_buffer_size_in_bytes, cudaMemcpyDeviceToHost));

    std::fill(host_result_vector_total.begin(), host_result_vector_total.end(), 0);

    for (int i = 0; i < threadsPerBlock; ++i) {
      host_result_vector_total['A' - 'A'] += host_result_vector[i * CACHE_LINE_SIZE + 'A' - 'A'];
      host_result_vector_total['C' - 'A'] += host_result_vector[i * CACHE_LINE_SIZE + 'A' - 'A'];
      host_result_vector_total['G' - 'A'] += host_result_vector[i * CACHE_LINE_SIZE + 'A' - 'A'];
      host_result_vector_total['N' - 'A'] += host_result_vector[i * CACHE_LINE_SIZE + 'A' - 'A'];
      host_result_vector_total['T' - 'A'] += host_result_vector[i * CACHE_LINE_SIZE + 'A' - 'A'];
    }

    cuda_stopwatch.stop();

    // Check results.
    validator::validate_results(host_result_vector_total);
  }
}
