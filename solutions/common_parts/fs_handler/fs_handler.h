#ifndef COMMON_PARTS_FS_HANDLER_H
#define COMMON_PARTS_FS_HANDLER_H

#include <fstream>
#include <iostream>
#include <type_traits>

namespace fs_handler {
  template<typename S>
  auto get_filesize(S &&filepath, std::ifstream &file) {
    if (!file.is_open()) {
      std::cerr << "Could not open the file - '" << filepath << "'" << std::endl;
      exit(EXIT_FAILURE);
    }
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    return size;
  }

  template<typename S>
  auto get_filesize(S &&filepath) {
    std::ifstream file(filepath, std::ios::binary);
    auto size = get_filesize(std::forward<S>(filepath), file);
    file.close();

    return size;
  }

  template<typename D>
  auto read_file(std::ifstream &file, D *buffer, size_t size) {
    return file.read(buffer, size) ? buffer : nullptr;
  }

  template<typename S>
  auto read_file(S &&filepath) {
    std::ifstream file(filepath, std::ios::binary);

    const auto size = get_filesize(std::forward<S>(filepath), file);

    using BASE = std::remove_reference_t<std::remove_cv_t<S>>;

    BASE buffer(size, '\0');
    if (read_file(file, const_cast<char *>(buffer.data()), size) == nullptr) {
      file.close();
      return BASE{};
    }

    file.close();
    return buffer;
  }

  template<typename S, typename D>
  auto read_file(S &&filepath, D *buffer, size_t size) {
    std::ifstream file(filepath, std::ios::binary);
    auto read_buffer = read_file(file, buffer, size);
    file.close();

    return read_buffer;
  }

}

#endif //COMMON_PARTS_FS_HANDLER_H
