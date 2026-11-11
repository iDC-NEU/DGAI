#pragma once
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <stdexcept>
#include "utils/log.h"
namespace pipeann {
// Admission gate + active outer guards: no thread slots, forced timeout, or ABA
// between index instances. Readers take no mutex but wait during reclamation.
class ReaderEpoch {
  std::atomic<bool> closed_{false};
  std::atomic<uint64_t> readers_{0};
  std::mutex writer_;
  static auto &depths() {
    thread_local std::unordered_map<ReaderEpoch *, unsigned> d; return d;
  }
 public:
  ReaderEpoch() = default;
  ReaderEpoch(const ReaderEpoch &) = delete;
  ReaderEpoch &operator=(const ReaderEpoch &) = delete;
  void enter() {
    auto &d = depths()[this]; if (d++) return;
    for (;;) {
      while (closed_.load(std::memory_order_seq_cst)) std::this_thread::yield();
      readers_.fetch_add(1, std::memory_order_seq_cst);
      if (!closed_.load(std::memory_order_seq_cst)) return;
      readers_.fetch_sub(1, std::memory_order_seq_cst);
    }
  }
  void leave() { auto &d = depths().at(this); if (--d == 0) readers_.fetch_sub(1, std::memory_order_seq_cst); }
  template<class F> void quiescent(F f, uint64_t warn_ms = 2000) {
    if (depths()[this]) throw std::logic_error("reclamation inside reader guard");
    std::lock_guard<std::mutex> lock(writer_); closed_.store(true, std::memory_order_seq_cst);
    auto last = std::chrono::steady_clock::now();
    while (readers_.load(std::memory_order_seq_cst)) {
      auto now = std::chrono::steady_clock::now();
      if (std::chrono::duration_cast<std::chrono::milliseconds>(now-last).count() >= (long long)warn_ms) {
        LOG(WARNING) << "ReaderEpoch waiting for " << readers_.load() << " readers; reclamation deferred"; last = now;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    try { f(); } catch (...) { closed_.store(false, std::memory_order_seq_cst); throw; }
    closed_.store(false, std::memory_order_seq_cst);
  }
  void synchronize(uint64_t timeout_ms=2000) { quiescent([]{}, timeout_ms); }
};
struct EpochGuard {
  ReaderEpoch &ep;
  explicit EpochGuard(ReaderEpoch &e): ep(e) { ep.enter(); }
  ~EpochGuard() { ep.leave(); }
  EpochGuard(const EpochGuard &) = delete;
  EpochGuard &operator=(const EpochGuard &) = delete;
};
}
