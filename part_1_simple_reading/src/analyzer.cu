#include "../inc/analyzer.cuh"

__global__ void analyzeGenome(
    const char *device_genome_buffer, std::uint32_t *results_vector, std::size_t genome_size
) {
  unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= genome_size) return;

  const std::uint8_t nucleotide = device_genome_buffer[i];

  constexpr std::uint8_t charACode = 'A';

  atomicAdd(results_vector + nucleotide - charACode, 1);
}


void run_analyzer(ThreadSafeQueue<std::string> &genomes_queue) {

  constexpr std::uint32_t expected_genome_size = 100 * 1024 * 1024; // 100 MB
  constexpr int threadsPerBlock = 256;
  constexpr int blocksPerGrid = (expected_genome_size + threadsPerBlock - 1) / threadsPerBlock;

  cudaDeviceProp prop{};
  checkCuda(cudaGetDeviceProperties(&prop, 0));
  std::cout << "Device: " << prop.name << std::endl;

  // Allocate genome buffer on device.
  char *device_genome_buffer;

  checkCuda(cudaMalloc(&device_genome_buffer, expected_genome_size)); // device

  // Allocate result vector on device.
  std::uint32_t *results_vector;

  constexpr std::uint32_t results_buffer_size = 16;
  constexpr std::uint32_t results_buffer_size_in_bytes = results_buffer_size * sizeof(std::uint32_t);

  checkCuda(cudaMalloc(&results_vector, results_buffer_size_in_bytes)); // device

  std::vector<std::uint32_t> host_result_vector(results_buffer_size);

  // Events for timing.
  cudaEvent_t startEvent, stopEvent;

  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));


  while (true) {
    auto file = genomes_queue.dequeue();

    // Catch poison pill.
    if (file.empty()) break;

    printf("Transfer size (MB): %lu\n", file.size() / (1024 * 1024));

    checkCuda(cudaEventRecord(startEvent, nullptr));

    checkCuda(cudaMemset(results_vector, 0, results_buffer_size_in_bytes));

    checkCuda(cudaMemcpy(device_genome_buffer, file.data(), file.size(), cudaMemcpyHostToDevice));

    // Invoke kernel.
    analyzeGenome<<<blocksPerGrid, threadsPerBlock>>>(device_genome_buffer, results_vector, file.size());

    checkCuda(cudaEventSynchronize(stopEvent));

    checkCuda(
        cudaMemcpy(
            host_result_vector.data(), results_vector, results_buffer_size_in_bytes, cudaMemcpyDeviceToHost
        )
    );

    checkCuda(cudaEventRecord(stopEvent, nullptr));
    checkCuda(cudaEventSynchronize(stopEvent));

    // Check results.
    check_results(host_result_vector);

    float time;
    checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));
    printf("  Host to Device bandwidth (GB/s): %f\n", static_cast<double>(file.size()) * 1e-6 / time);
  }

  // Clean up events.
  checkCuda(cudaEventDestroy(startEvent));
  checkCuda(cudaEventDestroy(stopEvent));

  cudaFree(device_genome_buffer);
}