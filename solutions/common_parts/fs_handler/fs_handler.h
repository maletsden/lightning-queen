#ifndef COMMON_PARTS_FS_HANDLER_H
#define COMMON_PARTS_FS_HANDLER_H

#include <fstream>
#include <iostream>

namespace fs_handler {
  template <typename S>
  S read_file(S&& filepath) {

    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      std::cerr << "Could not open the file - '" << filepath << "'" << std::endl;
      exit(EXIT_FAILURE);
    }
    file.seekg(0, std::ios::end);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    S buffer(size, '0');
    if (file.read(const_cast<char *>(buffer.data()), size)) {
      return buffer;
    }

    return S();
  }
}

#endif //COMMON_PARTS_FS_HANDLER_H
