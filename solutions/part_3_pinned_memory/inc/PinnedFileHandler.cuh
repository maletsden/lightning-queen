#ifndef PART_3_PINNED_MEMORY_PINNED_FILE_HANDLER_H
#define PART_3_PINNED_MEMORY_PINNED_FILE_HANDLER_H

#include <cstddef>

class PinnedFileHandler {
public:
  PinnedFileHandler() = default;

  explicit PinnedFileHandler(size_t filesize);

  ~PinnedFileHandler();

  PinnedFileHandler(const PinnedFileHandler &) noexcept = default;

  PinnedFileHandler &operator=(const PinnedFileHandler &) noexcept = default;

  PinnedFileHandler(PinnedFileHandler &&) noexcept = default;

  PinnedFileHandler &operator=(PinnedFileHandler &&) noexcept = default;

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


#endif //PART_3_PINNED_MEMORY_PINNED_FILE_HANDLER_H

