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


// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n",
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

std::string read_file(const std::string &filepath) {

  std::ifstream file(filepath, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::string buffer(size, '0');
  if (file.read(const_cast<char *>(buffer.data()), size)) {
    return buffer;
  }

  return "";
}

void check_results(const std::vector<std::uint32_t>& results) {
  const auto res = std::accumulate(results.begin(), results.end(), std::uint32_t(0));
  constexpr std::uint32_t expected_res = 100 * 1024 * 1024; // 100 MB
  assert(res == expected_res);

  constexpr std::uint32_t expected_res_per_nucleotide = expected_res / 5; // 20 MB

  assert(results['A' - 'A'] == expected_res_per_nucleotide);
  assert(results['C' - 'A'] == expected_res_per_nucleotide);
  assert(results['G' - 'A'] == expected_res_per_nucleotide);
  assert(results['N' - 'A'] == expected_res_per_nucleotide);
  assert(results['T' - 'A'] == expected_res_per_nucleotide);
}

void run_analyzer() {

  constexpr std::uint32_t expected_genome_size = 100 * 1024 * 1024; // 100 MB
  constexpr int threadsPerBlock = 256;
  constexpr int blocksPerGrid = (expected_genome_size + threadsPerBlock - 1) / threadsPerBlock;
  constexpr auto genomes_directory = "../data/";
  constexpr auto genomes_paths_file_path = "../data/genomes_paths.txt";

  ThreadSafeQueue<std::string> genomes_queue{};


  // Producer.
  std::thread producer{[&genomes_queue]() {
    std::ifstream genomes_paths_file(genomes_paths_file_path);

    std::string genome_path;
    while (std::getline(genomes_paths_file, genome_path)) {

      auto relative_path = genomes_directory + genome_path;
      auto file = read_file(relative_path);

      if (file.empty()) {
        std::cerr << "Failed reading file: " << genome_path << std::endl;
        continue;
      }

      genomes_queue.enqueue(file);

    }

    // Add poison pill.
    genomes_queue.enqueue("");
  }};



  // Consumer.
  {

    cudaDeviceProp prop{};
    checkCuda(cudaGetDeviceProperties(&prop, 0));
    printf("\nDevice: %s\n", prop.name);


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

  producer.join();

}