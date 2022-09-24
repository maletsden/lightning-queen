#ifndef COMMON_PARTS_FILE_HANDLER_H
#define COMMON_PARTS_FILE_HANDLER_H

#include <string>

#include "fs_type_traits.h"

namespace fs_handler {
  template<typename stream_type, typename S>
  auto assert_file_is_open(const stream_type &file, S &&msg) {
    if (!file.is_open()) {
      std::cerr << msg << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  template<typename T>
  class FileHandler {
  public:
    template<typename S>
    explicit FileHandler(const S &filepath, std::ios_base::openmode mode = std::ios_base::in) {
      file.open(filepath, mode);

      assert_file_is_open(file, "ERROR: Cannot open file " + filepath);
    }

    ~FileHandler() {
      file.close();
    }

    T file;
  };

  inline auto make_readable_file_handler(const std::string &filepath) {
    return FileHandler<std::ifstream>{filepath};
  }

  inline auto make_readable_binary_file_handler(const std::string &filepath) {
    return FileHandler<std::ifstream>{filepath, std::ios_base::in | std::ios_base::binary};
  }

  inline auto make_writable_file_handler(const std::string &filepath) {
    return FileHandler<std::ofstream>{filepath, std::ios_base::out};
  }

  inline auto make_appendable_file_handler(const std::string &filepath) {
    return FileHandler<std::ofstream>{filepath, std::ios_base::app};
  }
}


#endif //COMMON_PARTS_FILE_HANDLER_H
