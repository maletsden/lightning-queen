#include "analyzer.cuh"

#include <iostream>
#include <vector>

#include <validator/validator.h>
#include <cuda_validator/cuda_validator.h>
#include <cuda_stopwatch/CudaStopwatch.cuh>

__global__ void analyzer::analyzeGenome(
    const char *device_genome_buffer, RESULT_T *results_vector, std::size_t genome_size
) {
  const auto i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= genome_size) return;

  const std::uint8_t nucleotide = device_genome_buffer[i];

  constexpr std::uint8_t charACode = 'A';

  atomicAdd(results_vector + nucleotide - charACode, 1);
}


void analyzer::analyze(ThreadSafeQueue<std::string> &genomes_queue) {
  constexpr RESULT_T expected_genome_size = 100 * 1024 * 1024; // 100 MB
  constexpr int threadsPerBlock = 256;
  constexpr int blocksPerGrid = (expected_genome_size + threadsPerBlock - 1) / threadsPerBlock;

  cudaDeviceProp prop{};
  cuda_validator::check_error(cudaGetDeviceProperties(&prop, 0));
  std::cout << "Device: " << prop.name << std::endl;

  // Allocate genome buffer on device.
  char *device_genome_buffer;

  cuda_validator::check_error(cudaMalloc(&device_genome_buffer, expected_genome_size)); // device

  // Allocate result vector on device.
  RESULT_T *results_vector;

  constexpr RESULT_T results_buffer_size = 20;
  constexpr RESULT_T results_buffer_size_in_bytes = results_buffer_size * sizeof(RESULT_T);

  cuda_validator::check_error(cudaMalloc(&results_vector, results_buffer_size_in_bytes)); // device

  std::vector<RESULT_T> host_result_vector(results_buffer_size);

  // Events for timing.
  cudaEvent_t startEvent, stopEvent;

  cuda_validator::check_error(cudaEventCreate(&startEvent));
  cuda_validator::check_error(cudaEventCreate(&stopEvent));

  CudaStopwatch cuda_stopwatch;

  while (true) {
    const auto file = genomes_queue.dequeue();

    // Catch poison pill.
    if (file.empty()) break;

    cuda_stopwatch.start(file.size());

    cuda_validator::check_error(cudaMemset(results_vector, 0, results_buffer_size_in_bytes));

    cuda_validator::check_error(cudaMemcpy(device_genome_buffer, file.data(), file.size(), cudaMemcpyHostToDevice));

    // Invoke kernel.
    analyzeGenome<<<blocksPerGrid, threadsPerBlock>>>(device_genome_buffer, results_vector, file.size());

    cuda_validator::check_error(
        cudaMemcpy(
            host_result_vector.data(), results_vector, results_buffer_size_in_bytes, cudaMemcpyDeviceToHost
        )
    );

    cuda_stopwatch.stop();

    // Check results.
    validator::validate_results(host_result_vector);
  }

  cudaFree(device_genome_buffer);
}