#ifndef GENOME_ANALYZER_GENOME_ZIPPER_H
#define GENOME_ANALYZER_GENOME_ZIPPER_H

#include <string>
#include <array>

namespace genome_zipper {
  using BYTE = std::uint8_t;

  struct ZippedGenome {
    size_t real_size{0};
    std::string container;
  };

  constexpr std::array<BYTE, 'T' - 'A' + 1> encodeChar = {
      0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 3
  };
  constexpr std::array<char, 4> decodeChar = {
      'A', 'C', 'G', 'T'
  };


  char zip_3_chars(const char *chars);

  void unzip_3_chars(char zipped_chars, std::string &result);

  auto zip(const char *genome, size_t genome_size) {
    ZippedGenome zipped_genome;
    zipped_genome.real_size = genome_size;
    zipped_genome.container.reserve((genome_size + 2) / 3);

    const auto remainder = genome_size % 3;
    const auto genome_end = genome + genome_size - remainder;

    for (const char *genome_ptr = genome; genome_ptr < genome_end; genome_ptr += 3) {
      zipped_genome.container.push_back(zip_3_chars(genome_ptr));
    }

    // add additional fictional 'A' to simulate genome size dividable by 3
    if (remainder) {
      char remaining[3] = {'A', 'A', 'A'};

      for (auto i = 0; i < remainder; ++i) {
        remaining[i] = *(genome_end + i);
      }

      zipped_genome.container.push_back(zip_3_chars(remaining));
    }

    return zipped_genome;
  }

  template<typename S>
  auto zip(S &&genome) {
    return zip(genome.data(), genome.size());
  }

  template<typename S, int size>
  auto zip(S(&genome)[size]) {
    return zip(genome, size - (genome[size - 1] == '\0'));
  }

  char zip_3_chars(const char *chars) {
    const BYTE N_num = (chars[0] == 'N') + (chars[1] == 'N') + (chars[2] == 'N');

    BYTE encoded_char = 0;

    encoded_char |= N_num;
    encoded_char <<= 2;


    switch (N_num) {
      case 0:
        encoded_char |= encodeChar[chars[0] - 'A'];
        encoded_char <<= 2;

        encoded_char |= encodeChar[chars[1] - 'A'];
        encoded_char <<= 2;

        encoded_char |= encodeChar[chars[2] - 'A'];
        break;
      case 1: {
        // in this case we also need to save index of N
        const BYTE N_index = 0 + (chars[1] == 'N') + ((chars[2] == 'N') << 1);

        encoded_char |= N_index;

        // encode 2 "not N" chars
        for (auto i = 0; i < 3; ++i) {
          if (i == N_index) continue;
          encoded_char <<= 2;
          encoded_char |= encodeChar[chars[i] - 'A'];
        }
        break;
      }
      case 2: {
        // in these case we need to save index of non-N and save it
        const BYTE not_N_index = 0 + (chars[1] != 'N') + ((chars[2] != 'N') << 1);

        encoded_char |= not_N_index;
        encoded_char <<= 2;
        encoded_char |= encodeChar[chars[not_N_index] - 'A'];
        encoded_char <<= 2;
        break;
      }
      default:
        // in other case (N_num == 3) we can just finish
        encoded_char <<= 4;
        break;
    }
    return static_cast<char>(encoded_char);
  }

  auto unzip(const char *zipped_genome, size_t real_genome_size) {
    std::string genome;

    const auto zipped_genome_size = (real_genome_size + 2) / 3;

    for (auto i = 0; i < zipped_genome_size; ++i) {
      unzip_3_chars(zipped_genome[i], genome);
    }

    genome.resize(real_genome_size);

    return genome;
  }

  auto unzip(const ZippedGenome &zipped_genome) {
    return unzip(zipped_genome.container.data(), zipped_genome.real_size);
  }

  void unzip_3_chars(char zipped_chars, std::string &result) {
    constexpr auto first_2_bits = 0b11000000;
    constexpr auto second_2_bits = 0b00110000;
    constexpr auto third_2_bits = 0b00001100;
    constexpr auto fourth_2_bits = 0b00000011;

    const BYTE N_num = zipped_chars & first_2_bits;

    switch (N_num) {
      case 0b00000000:
        result.push_back(decodeChar[(zipped_chars & second_2_bits) >> 4]);
        result.push_back(decodeChar[(zipped_chars & third_2_bits) >> 2]);
        result.push_back(decodeChar[zipped_chars & fourth_2_bits]);
        break;
      case 0b01000000: {
        // in this case we also need to save index of N
        const BYTE N_index = (zipped_chars & second_2_bits) >> 4;

        // decode 2 "not N" chars
        char decoded[2] = {decodeChar[(zipped_chars & third_2_bits) >> 2], decodeChar[zipped_chars & fourth_2_bits]};
        auto decoded_i = 0;

        result.push_back(0 == N_index ? 'N' : decoded[decoded_i++]);
        result.push_back(1 == N_index ? 'N' : decoded[decoded_i++]);
        result.push_back(2 == N_index ? 'N' : decoded[decoded_i]);
        break;
      }
      case 0b10000000: {
        // in these case we need to save index of non-N and save it
        const BYTE not_N_index = (zipped_chars & second_2_bits) >> 4;

        // decode "not N" char
        const char decoded = decodeChar[(zipped_chars & third_2_bits) >> 2];

        result.push_back(0 == not_N_index ? decoded : 'N');
        result.push_back(1 == not_N_index ? decoded : 'N');
        result.push_back(2 == not_N_index ? decoded : 'N');
        break;
      }
      default:
        // in other case (N_num == 3) we can just add them
        result.push_back('N');
        result.push_back('N');
        result.push_back('N');
        break;
    }
  }

}

#endif //GENOME_ANALYZER_GENOME_ZIPPER_H
