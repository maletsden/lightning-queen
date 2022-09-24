#ifndef GENOME_ANALYZER_ZIPPED_GENOME_H
#define GENOME_ANALYZER_ZIPPED_GENOME_H

#include <string>

namespace genome_zipper {
  template<typename Container>
  class ZippedGenome {
  public:
    [[nodiscard]] bool is_empty() const {
      return real_size == 0;
    }

    size_t real_size{0};
    Container container;
  };
}

#endif //GENOME_ANALYZER_ZIPPED_GENOME_H
