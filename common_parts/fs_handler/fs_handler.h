#ifndef COMMON_PARTS_FS_HANDLER_H
#define COMMON_PARTS_FS_HANDLER_H

#include <fstream>
#include <iostream>

#include "FileHandler.h"
#include "fs_type_traits.h"

namespace fs_handler {
  template<typename stream_type, std::enable_if_t<is_input_file_stream<stream_type>, bool> = true>
  auto get_filesize(stream_type &&file) {
    assert_file_is_open(file, "File is not opened.");

    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    return size;
  }

  auto get_filesize(const std::string &filepath) {
    return get_filesize(make_readable_file_handler(filepath).file);
  }

  template<typename T, std::enable_if_t<is_input_file_stream<T>, bool> = true>
  auto get_filesize(FileHandler<T> &file_handler) {
    return get_filesize(file_handler.file);
  }

  template<typename stream_type, std::enable_if_t<is_input_file_stream<stream_type>, bool> = true>
  auto get_filesize_till_end(stream_type &&file) {
    const std::streampos start = file.tellg();

    file.seekg(0, std::ios::end);
    const std::streampos end = file.tellg();
    file.seekg(start);

    return end - start;
  }

  template<typename T, std::enable_if_t<is_input_file_stream<T>, bool> = true>
  auto get_filesize_till_end(FileHandler<T> &file_handler) {
    return get_filesize_till_end(file_handler.file);
  }

  template<typename stream_type, typename D, std::enable_if_t<is_input_file_stream<stream_type>, bool> = true>
  auto read_file(stream_type &&file, D *buffer, size_t size) {
    assert_file_is_open(file, "File is not opened.");

    return file.read(buffer, size) ? buffer : nullptr;
  }

  template<typename S>
  auto read_file(S &&filepath) {
    auto file_handler = make_readable_binary_file_handler(std::forward<S>(filepath));
    const auto size = get_filesize(file_handler.file);

    std::string buffer(size, '\0');
    if (read_file(file_handler.file, const_cast<char *>(buffer.data()), size) == nullptr) {
      return decltype(buffer){};
    }

    return buffer;
  }

  template<typename D>
  auto read_file(const std::string &filepath, D *buffer, size_t size) {
    return read_file(make_readable_binary_file_handler(filepath).file, buffer, size);
  }

  template<typename S, typename D>
  auto write_file(S &&filepath, D &&data) {
    make_writable_file_handler(std::forward<S>(filepath)) << std::forward<D>(data);
  }

  template<typename T, typename D>
  auto &operator<<(FileHandler<T> &file_handler, D &&data) {
    file_handler.file << std::forward<D>(data);
    return file_handler;
  }

  template<typename T, typename D>
  auto &operator<<(FileHandler<T> &&file_handler, D &&data) {
    file_handler.file << std::forward<D>(data);
    return file_handler;
  }

  template<typename T, std::enable_if_t<is_input_file_stream<T>, bool> = true>
  auto &getline(FileHandler<T> &file_handler, std::string &str) {
    std::getline(file_handler.file, str);
    return file_handler;
  }

}

#endif //COMMON_PARTS_FS_HANDLER_H
