#ifndef COMMON_PARTS_FS_HANDLER_H
#define COMMON_PARTS_FS_HANDLER_H

#include <fstream>
#include <iostream>
#include <type_traits>

#include "FileHandler.h"

namespace fs_handler {
  template<class T>
  struct remove_cvref {
    typedef std::remove_cv_t<std::remove_reference_t<T>> type;
  };
  template<class T>
  using remove_cvref_t = typename remove_cvref<T>::type;

  template<typename stream_type>
  constexpr bool is_file_stream = std::is_base_of_v<std::fstream,  remove_cvref_t<stream_type>>;
  template<typename stream_type>
  constexpr bool is_input_file_stream =
      std::is_base_of_v<std::ifstream,  remove_cvref_t<stream_type>> || is_file_stream<stream_type>;
  template<typename stream_type>
  constexpr bool is_output_file_stream =
      std::is_base_of_v<std::ofstream, remove_cvref_t<stream_type>> || is_file_stream<stream_type>;


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

  template<typename stream_type, typename D, std::enable_if_t<is_input_file_stream<stream_type>, bool> = true>
  auto read_file(stream_type &&file, D *buffer, size_t size) {
    assert_file_is_open(file, "File is not opened.");

    return file.read(buffer, size) ? buffer : nullptr;
  }

  template<typename S>
  auto read_file(S &&filepath) {
    auto file_handler = make_readable_file_handler(std::forward<S>(filepath));
    const auto size = get_filesize(file_handler.file);

    std::string buffer(size, '\0');
    if (read_file(file_handler.file, const_cast<char *>(buffer.data()), size) == nullptr) {
      return decltype(buffer){};
    }

    return buffer;
  }

  template<typename D>
  auto read_file(const std::string &filepath, D *buffer, size_t size) {
    return read_file(make_readable_file_handler(filepath).file, buffer, size);
  }

  template<typename S, typename D>
  auto write_file(S &&filepath, D &&data) {
    make_writable_file_handler(std::forward<S>(filepath)) << std::forward<D>(data);
  }
}

#endif //COMMON_PARTS_FS_HANDLER_H
