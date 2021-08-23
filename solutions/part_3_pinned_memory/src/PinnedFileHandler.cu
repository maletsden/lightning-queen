#include "PinnedFileHandler.cuh"

#include <cuda.h>
#include <cuda_validator/cuda_validator.h>

PinnedFileHandler::PinnedFileHandler(size_t filesize) : size(filesize) {
  cuda_validator::check_error(cudaHostAlloc(&data, filesize, cudaHostAllocDefault));
}

PinnedFileHandler::~PinnedFileHandler() {
  if (data) cudaFreeHost(data);
}
