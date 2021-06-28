#ifndef PART_1_SIMPLE_READING_ANALYZER_CUH
#define PART_1_SIMPLE_READING_ANALYZER_CUH

#include <cstdio>
#include <iostream>
#include <vector>
#include <array>
#include <numeric>
#include <thread>
#include <fstream>
#include <cstdio>
#include <cassert>
#include <cuda.h>
#include <driver_types.h>

#include "ThreadSafeQueue.h"
#include "helpers.h"


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

void run_analyzer(ThreadSafeQueue<std::string>& genomes_queue);

#endif //PART_1_SIMPLE_READING_ANALYZER_CUH