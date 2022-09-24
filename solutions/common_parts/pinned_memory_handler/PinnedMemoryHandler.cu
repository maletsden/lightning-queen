#include "PinnedMemoryHandler.cuh"

#include <cuda_runtime.h>
#include <cuda_validator/cuda_validator.h>
#include <stdexcept>

PinnedMemoryHandler::PinnedMemoryHandler(size_t size, unsigned int flags) : m_size(size) {
  cuda_validator::check_error(cudaHostAlloc(&m_data, size, flags));
  if (!m_data) {
    throw std::runtime_error("Failed allocating pinned memory.");
  }
}

PinnedMemoryHandler::PinnedMemoryHandler(size_t size, char *data) : m_size(size), m_data(data) {}

PinnedMemoryHandler::~PinnedMemoryHandler() {
  if (m_data) cudaFreeHost(m_data);
}

