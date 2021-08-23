#include "PinnedMemoryHandler.cuh"

#include <cuda.h>
#include <cuda_validator/cuda_validator.h>

PinnedMemoryHandler::PinnedMemoryHandler(size_t filesize, unsigned int flags) : size(filesize) {
  cuda_validator::check_error(cudaHostAlloc(&data, filesize, flags));
}

PinnedMemoryHandler::~PinnedMemoryHandler() {
  if (data) cudaFreeHost(data);
}
