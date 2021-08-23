#ifndef COMMON_PARTS_PINNED_MEMORY_HANDLER_H
#define COMMON_PARTS_PINNED_MEMORY_HANDLER_H

#include <cstddef>

#include <driver_types.h>

class PinnedMemoryHandler {
public:
  PinnedMemoryHandler() = default;

  explicit PinnedMemoryHandler(size_t filesize, unsigned int flags = cudaHostAllocDefault);

  ~PinnedMemoryHandler();

  PinnedMemoryHandler(const PinnedMemoryHandler &) noexcept = default;

  PinnedMemoryHandler &operator=(const PinnedMemoryHandler &) noexcept = default;

  PinnedMemoryHandler(PinnedMemoryHandler &&) noexcept = default;

  PinnedMemoryHandler &operator=(PinnedMemoryHandler &&) noexcept = default;

  [[nodiscard]] auto get_data() const noexcept {
    return data;
  }

  [[nodiscard]] auto get_size() const noexcept {
    return size;
  }

  [[nodiscard]] bool is_empty() const noexcept {
    return size == 0;
  }

private:
  char *data{nullptr};
  size_t size{0};
};


#endif //COMMON_PARTS_PINNED_MEMORY_HANDLER_H

