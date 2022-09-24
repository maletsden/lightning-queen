#ifndef COMMON_PARTS_PINNED_MEMORY_HANDLER_H
#define COMMON_PARTS_PINNED_MEMORY_HANDLER_H

#include <cstddef>

#include <driver_types.h>

class PinnedMemoryHandler {
public:
  PinnedMemoryHandler() = default;

  explicit PinnedMemoryHandler(size_t size, unsigned int flags = cudaHostAllocDefault);
  explicit PinnedMemoryHandler(size_t size, char *data);

  ~PinnedMemoryHandler();

  PinnedMemoryHandler(const PinnedMemoryHandler &) noexcept = default;

  PinnedMemoryHandler &operator=(const PinnedMemoryHandler &) noexcept = default;

  PinnedMemoryHandler(PinnedMemoryHandler &&) noexcept = default;

  PinnedMemoryHandler &operator=(PinnedMemoryHandler &&) noexcept = default;

  [[nodiscard]] auto get_data() const noexcept {
    return m_data;
  }

  [[nodiscard]] auto get_size() const noexcept {
    return m_size;
  }

  [[nodiscard]] bool is_empty() const noexcept {
    return m_size == 0;
  }

private:
  char *m_data{nullptr};
  size_t m_size{0};
};


#endif //COMMON_PARTS_PINNED_MEMORY_HANDLER_H

