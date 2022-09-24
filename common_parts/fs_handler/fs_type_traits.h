#ifndef COMMON_PARTS_FS_TYPE_TRAITS_H
#define COMMON_PARTS_FS_TYPE_TRAITS_H

#include <type_traits>

namespace fs_handler {
  template<class T>
  struct remove_cvref {
    typedef std::remove_cv_t<std::remove_reference_t<T>> type;
  };
  template<class T>
  using remove_cvref_t = typename remove_cvref<T>::type;

  template<typename stream_type>
  constexpr bool is_file_stream = std::is_base_of_v<std::fstream, remove_cvref_t<stream_type>>;
  template<typename stream_type>
  constexpr bool is_input_file_stream =
      std::is_base_of_v<std::ifstream, remove_cvref_t<stream_type>> || is_file_stream<stream_type>;
  template<typename stream_type>
  constexpr bool is_output_file_stream =
      std::is_base_of_v<std::ofstream, remove_cvref_t<stream_type>> || is_file_stream<stream_type>;
}

#endif //COMMON_PARTS_FS_TYPE_TRAITS_H
