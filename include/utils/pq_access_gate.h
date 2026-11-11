#pragma once
#include <array>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <thread>
namespace pipeann {
// Writer-preferred admission with independent reader cache lines. Readers of
// disk PQ still take a shard read lock; MemNav's private PQ takes none.
// A waiting writer closes admission before draining the shards, preventing
// starvation without one globally contended reader counter per PQ operation.
class PQAccessGate {
  static constexpr size_t shards=32;
  struct alignas(64) Shard {std::shared_mutex mutex;};
  std::array<Shard,shards> slots_;
  std::atomic<bool> closed_{false};
  std::mutex writer_mu_;
  static size_t bucket() {
    static std::atomic<size_t> ticket{0};
    thread_local const size_t b=ticket.fetch_add(1,std::memory_order_relaxed)%shards;
    return b;
  }
 public:
  void lock_shared() {
    auto &m=slots_[bucket()].mutex;
    for(;;){
      while(closed_.load(std::memory_order_seq_cst))std::this_thread::yield();
      m.lock_shared();
      if(!closed_.load(std::memory_order_seq_cst))return;
      m.unlock_shared();
    }
  }
  void unlock_shared(){slots_[bucket()].mutex.unlock_shared();}
  void lock(){
    writer_mu_.lock();closed_.store(true,std::memory_order_seq_cst);
    for(auto &s:slots_)s.mutex.lock();
  }
  void unlock(){
    for(auto &s:slots_)s.mutex.unlock();
    closed_.store(false,std::memory_order_seq_cst);writer_mu_.unlock();
  }
};
}
