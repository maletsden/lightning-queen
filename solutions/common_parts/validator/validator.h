#ifndef COMMON_PARTS_VALIDATOR_H
#define COMMON_PARTS_VALIDATOR_H

#include <vector>
#include <numeric>
#include <cassert>
#include <type_traits>

namespace validator {
  constexpr auto ExpectedResult = 100 * 1024 * 1024; // 100 MB
  constexpr auto ExpectedNucleotidesNum = 5;
  constexpr auto ExpectedResultPerNucleotide = ExpectedResult / ExpectedNucleotidesNum; // 20 MB

  constexpr auto SetRedColor = "\033[31m";
  constexpr auto ResetAllStyles = "\033[0m";

  template<typename T, typename D>
  constexpr bool are_integral_v = std::is_integral_v<T> && std::is_integral_v<D>;

  template<typename T, typename D, std::enable_if_t<are_integral_v<T, D>, bool> = true>
  constexpr bool val_assert(const T result, const D expected_result, const std::string &msg = "") noexcept {
    if (result == expected_result) return true;

    std::cout << SetRedColor << "Validation is failed. Result (" << result << ") is not equal expected result - "
              << expected_result << ". " << msg << ResetAllStyles << std::endl;

    return false;
  }

  template<typename T, typename D, std::enable_if_t<are_integral_v<T, D>, bool> = true>
  constexpr bool
  validate_result_vector_min_size(const T size, const D min_size, const std::string &msg = "") noexcept {
    if (size >= min_size) return true;
    std::cout << SetRedColor << "Validation is failed. Result vector size must be bigger than "
              << min_size << ". Current size is " << size << ". " << msg << ResetAllStyles << std::endl;

    return false;
  }

  template<typename T>
  constexpr void validate_results(const std::vector<T> &results) noexcept {
    constexpr auto minResultVectorSize = 'T' - 'A' + 1;

    if (!validate_result_vector_min_size(results.size(), minResultVectorSize)) {
      return;
    }

    val_assert(std::accumulate(results.cbegin(), results.cend(), T()), ExpectedResult);

    val_assert(results['A' - 'A'], ExpectedResultPerNucleotide);
    val_assert(results['C' - 'A'], ExpectedResultPerNucleotide);
    val_assert(results['G' - 'A'], ExpectedResultPerNucleotide);
    val_assert(results['N' - 'A'], ExpectedResultPerNucleotide);
    val_assert(results['T' - 'A'], ExpectedResultPerNucleotide);
  }

  template<typename T>
  constexpr void validate_results_packed(const std::vector<T> &results) noexcept {
    if (!validate_result_vector_min_size(results.size(), ExpectedNucleotidesNum)) {
      return;
    }

    val_assert(std::accumulate(results.cbegin(), results.cend(), T()), ExpectedResult);

    for (auto i = 0; i < ExpectedNucleotidesNum; ++i) {
      val_assert(results[i], ExpectedResultPerNucleotide);
    }
  }
}

#endif //COMMON_PARTS_VALIDATOR_H
