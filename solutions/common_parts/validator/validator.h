#ifndef COMMON_PARTS_VALIDATOR_H
#define COMMON_PARTS_VALIDATOR_H

#include <vector>
#include <numeric>
#include <cassert>

namespace validator {

  template<typename T>
  void validate_results(const std::vector<T> &results) {
    constexpr T expected_res = 100 * 1024 * 1024; // 100 MB

    const auto res = std::accumulate(results.cbegin(), results.cend(), T());
    assert(res == expected_res);

    constexpr auto expected_res_per_nucleotide = expected_res / 5; // 20 MB

    assert(results['A' - 'A'] == expected_res_per_nucleotide);
    assert(results['C' - 'A'] == expected_res_per_nucleotide);
    assert(results['G' - 'A'] == expected_res_per_nucleotide);
    assert(results['N' - 'A'] == expected_res_per_nucleotide);
    assert(results['T' - 'A'] == expected_res_per_nucleotide);
  }
}

#endif //COMMON_PARTS_VALIDATOR_H
