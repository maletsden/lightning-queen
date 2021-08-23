#include "PinnedMemoryHandler.cuh"

#include <cuda.h>
#include <cuda_validator/cuda_validator.h>

PinnedMemoryHandler::PinnedMemoryHandler(size_t filesize) : size(filesize) {
  cuda_validator::check_error(cudaHostAlloc(&data, filesize, cudaHostAllocDefault));
}

PinnedMemoryHandler::~PinnedMemoryHandler() {
  if (data) cudaFreeHost(data);
}
