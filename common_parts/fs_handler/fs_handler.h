#ifndef COMMON_PARTS_FS_HANDLER_H
#define COMMON_PARTS_FS_HANDLER_H

#include <fstream>
#include <iostream>
#include <type_traits>

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

  template<typename stream_type>
  auto get_filesize(stream_type &file) {
    assert_file_is_open(file, "File is not opened.");

    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    return size;
  }

  auto get_filesize(const std::string &filepath) {
    FileHandler<std::ifstream> file_handler{filepath, std::ios::binary};
    return get_filesize(file_handler.file);
  }

  template<typename stream_type,
      typename D,
      std::enable_if_t<
          std::is_base_of_v<std::ifstream, stream_type> || std::is_base_of_v<std::fstream, stream_type>, bool> = true
  >
  auto read_file(stream_type &file, D *buffer, size_t size) {
    assert_file_is_open(file, "File is not opened.");

    return file.read(buffer, size) ? buffer : nullptr;
  }

  template<typename S>
  auto read_file(S &&filepath) {
    FileHandler<std::ifstream> file_handler{filepath, std::ios::binary};

    const auto size = get_filesize(file_handler.file);

    std::string buffer(size, '\0');
    if (read_file(file_handler.file, const_cast<char *>(buffer.data()), size) == nullptr) {
      return decltype(buffer){};
    }

    return buffer;
  }

  template<typename S, typename D>
  auto read_file(S &&filepath, D *buffer, size_t size) {
    FileHandler<std::ifstream> file_handler{filepath, std::ios::binary};
    return read_file(file_handler.file, buffer, size);
  }

  template<typename S, typename D>
  auto write_file(S &&filepath, D &&data) {
    FileHandler<std::ofstream> file_handler{filepath};
    file_handler.file << data;
  }

  template<typename S, typename D>
  auto write_files(S &&filepath, const std::vector<D> &data) {
    FileHandler<std::ofstream> file_handler{filepath};
    for (const auto &str: data) file_handler.file << str;
  }

}

#endif //COMMON_PARTS_FS_HANDLER_H
