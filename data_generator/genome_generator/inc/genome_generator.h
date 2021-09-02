#ifndef DATA_GENERATOR_GENOME_GENERATOR_H
#define DATA_GENERATOR_GENOME_GENERATOR_H

#include <cstddef>
#include <vector>
#include <string>

namespace genome_generator {
  void generate(const std::vector<std::string> &genomes_paths, size_t genome_size);
}

#endif //DATA_GENERATOR_GENOME_GENERATOR_H
