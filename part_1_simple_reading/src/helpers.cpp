#include "../inc/helpers.h"

void check_results(const std::vector<std::uint32_t> &results) {
  const auto res = std::accumulate(results.begin(), results.end(), std::uint32_t(0));
  constexpr std::uint32_t expected_res = 100 * 1024 * 1024; // 100 MB
  assert(res == expected_res);

  constexpr std::uint32_t expected_res_per_nucleotide = expected_res / 5; // 20 MB

  assert(results['A' - 'A'] == expected_res_per_nucleotide);
  assert(results['C' - 'A'] == expected_res_per_nucleotide);
  assert(results['G' - 'A'] == expected_res_per_nucleotide);
  assert(results['N' - 'A'] == expected_res_per_nucleotide);
  assert(results['T' - 'A'] == expected_res_per_nucleotide);
}