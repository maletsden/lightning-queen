#ifndef COMMON_PARTS_CUDA_VALIDATOR_H
#define COMMON_PARTS_CUDA_VALIDATOR_H

#include <cstdio>
#include <cassert>

namespace cuda_validator {

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.

  inline cudaError_t check_error(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
      fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
      assert(result == cudaSuccess);
    }
#endif
    return result;
  }
}
#endif //COMMON_PARTS_CUDA_VALIDATOR_H
