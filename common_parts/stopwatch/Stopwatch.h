#ifndef GENOME_ANALYZER_STOPWATCH_H
#define GENOME_ANALYZER_STOPWATCH_H

#include <chrono>
#include <string>
#include <iostream>
#include <atomic>


class Stopwatch {
public:
  using TIME_POINT = decltype(std::chrono::high_resolution_clock::now());

  static inline TIME_POINT getCurrentTimeFenced() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto res_time = std::chrono::high_resolution_clock::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return res_time;
  }

  template<class D>
  static inline auto toUs(D &&d) {
    return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
  }

  explicit Stopwatch(std::string &&msg) : m_msg(std::move(msg)), m_start_time(getCurrentTimeFenced()) {
  }

  ~Stopwatch() {
    std::cout << m_msg << toUs((m_stopped ? m_end_time : getCurrentTimeFenced()) - m_start_time) << " μs." << std::endl;
  }

  void stop() {
    m_stopped = true;
    m_end_time = getCurrentTimeFenced();
  }


private:
  TIME_POINT m_start_time, m_end_time{};
  std::string m_msg;
  bool m_stopped = false;
};

#endif //GENOME_ANALYZER_STOPWATCH_H
