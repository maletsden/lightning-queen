#include "analyzer.cuh"

#include <iostream>
#include <vector>
#include <cassert>
#include <string>

#include <validator/validator.h>
#include <cuda_validator/cuda_validator.h>
#include <cuda_stopwatch/CudaStopwatch.cuh>


__global__ void analyzer::analyze_genome(
    const char *device_genome_buffer, RESULT_T *results_vector, std::size_t zipped_genome_size
) {
  const auto i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i >= zipped_genome_size) return;

  const char zipped_nucleotides = device_genome_buffer[i];

  constexpr char decodeChar[4] = {
      'A', 'C', 'G', 'T'
  };
  constexpr auto first_2_bits = 0b11000000;
  constexpr auto second_2_bits = 0b00110000;
  constexpr auto third_2_bits = 0b00001100;
  constexpr auto fourth_2_bits = 0b00000011;
  const auto thread_result_offset = results_vector + threadIdx.x * CACHE_LINE_SIZE;

  const std::uint8_t N_num = zipped_nucleotides & first_2_bits;

  switch (N_num) {
    case 0b00000000:
      atomicAdd(thread_result_offset + decodeChar[(zipped_nucleotides & second_2_bits) >> 4] - 'A', 1);
      atomicAdd(thread_result_offset + decodeChar[(zipped_nucleotides & third_2_bits) >> 2] - 'A', 1);
      atomicAdd(thread_result_offset + decodeChar[zipped_nucleotides & fourth_2_bits] - 'A', 1);
      break;
    case 0b01000000: {
      atomicAdd(thread_result_offset + 'N' - 'A', 1);

      // decode 2 "not N" chars
      atomicAdd(thread_result_offset + decodeChar[(zipped_nucleotides & third_2_bits) >> 2] - 'A', 1);
      atomicAdd(thread_result_offset + decodeChar[zipped_nucleotides & fourth_2_bits] - 'A', 1);
      break;
    }
    case 0b10000000: {
      atomicAdd(thread_result_offset + 'N' - 'A', 2);
      // decode "not N" char
      atomicAdd(thread_result_offset + decodeChar[(zipped_nucleotides & third_2_bits) >> 2] - 'A', 1);
      break;
    }
    default:
      // in other case (N_num == 3) we can just add them
      atomicAdd(thread_result_offset + 'N' - 'A', 3);
      break;
  }
}

std::vector<char> analyze_last_zipped_char(const char zipped_nucleotides) {
  constexpr char decodeChar[4] = {
      'A', 'C', 'G', 'T'
  };
  constexpr auto first_2_bits = 0b11000000;
  constexpr auto second_2_bits = 0b00110000;
  constexpr auto third_2_bits = 0b00001100;
  constexpr auto fourth_2_bits = 0b00000011;

  const std::uint8_t N_num = zipped_nucleotides & first_2_bits;

  switch (N_num) {
    case 0b00000000:
      return {
          decodeChar[(zipped_nucleotides & second_2_bits) >> 4],
          decodeChar[(zipped_nucleotides & third_2_bits) >> 2],
          decodeChar[(zipped_nucleotides & fourth_2_bits)],
      };
    case 0b01000000: {
      std::vector<char> decoded = {decodeChar[(zipped_nucleotides & third_2_bits) >> 2],
                                   decodeChar[zipped_nucleotides & fourth_2_bits]};
      const std::uint8_t N_index = (zipped_nucleotides & second_2_bits) >> 4;
      decoded.insert(decoded.begin() + N_index, 'N');
      return decoded;
    }
    case 0b10000000: {
      std::vector<char> decoded = {'N', 'N'};
      const std::uint8_t non_N_index = (zipped_nucleotides & second_2_bits) >> 4;
      decoded.insert(decoded.begin() + non_N_index, decodeChar[(zipped_nucleotides & third_2_bits) >> 2]);
      return decoded;
    }
    default:
      return {'N', 'N', 'N'};
  }
}

void analyzer::analyze(QUEUE_T &zipped_genomes_queue) {

  constexpr int threadsPerBlock = 256;
  constexpr int blocksPerGrid = (EXPECTED_GENOME_SIZE + threadsPerBlock - 1) / threadsPerBlock;
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
    auto zipped_genome_handler = zipped_genomes_queue.dequeue();

    // Catch poison pill.
    if (zipped_genome_handler.is_empty()) break;

    const auto filesize = zipped_genome_handler.real_size;
    std::cout << "Transfer m_size (MB): " << filesize / (1024 * 1024) << std::endl;

    cuda_stopwatch.start(filesize);

    cuda_validator::check_error(cudaMemset(results_vector, 0, results_buffer_size_in_bytes));

    char *device_genome_buffer;
    cuda_validator::check_error(
        cudaHostGetDevicePointer(&device_genome_buffer, zipped_genome_handler.container->get_data(), 0));

    // Invoke kernel.
    analyze_genome<<<blocksPerGrid, threadsPerBlock>>>(device_genome_buffer, results_vector,
                                                       zipped_genome_handler.container->get_size() - 1);


    cuda_validator::check_error(
        cudaMemcpy(host_result_vector.data(), results_vector, results_buffer_size_in_bytes, cudaMemcpyDeviceToHost));

    std::fill(host_result_vector_total.begin(), host_result_vector_total.end(), 0);

    for (int i = 0; i < threadsPerBlock; ++i) {
      host_result_vector_total['A' - 'A'] += host_result_vector[i * CACHE_LINE_SIZE + 'A' - 'A'];
      host_result_vector_total['C' - 'A'] += host_result_vector[i * CACHE_LINE_SIZE + 'C' - 'A'];
      host_result_vector_total['G' - 'A'] += host_result_vector[i * CACHE_LINE_SIZE + 'G' - 'A'];
      host_result_vector_total['N' - 'A'] += host_result_vector[i * CACHE_LINE_SIZE + 'N' - 'A'];
      host_result_vector_total['T' - 'A'] += host_result_vector[i * CACHE_LINE_SIZE + 'T' - 'A'];
    }

    const auto last_decode_char = analyze_last_zipped_char(
        zipped_genome_handler.container->get_data()[zipped_genome_handler.container->get_size() - 1]);
    for (int i = 0; i < filesize % 3; ++i) {
      ++host_result_vector_total[last_decode_char[i] - 'A'];
    }

    cuda_stopwatch.stop();

    // Check results.
    validator::validate_results(host_result_vector_total);
  }

  cudaFree(results_vector);
}
