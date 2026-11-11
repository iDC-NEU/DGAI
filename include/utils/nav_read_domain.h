#pragma once
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <utility>
#include <cstdint>
namespace pipeann {
// Reader admission never closes. A single maintenance writer publishes detached
// edges before rotating the generation and waits only for the prior cohort.
// The monotonic generation (not just a 1-bit toggle) rejects delayed entrants.
class NavReadDomain {
  struct alignas(64) Counter { std::atomic<uint64_t> value{0}; } readers_[2];
  alignas(64) std::atomic<uint64_t> generation_{0};
  std::mutex writer_;
 public:
  class Guard {
    NavReadDomain *owner_;uint64_t generation_;
   public:
    explicit Guard(NavReadDomain &d):owner_(&d) {
      for(;;){
        generation_=d.generation_.load(std::memory_order_seq_cst);
        auto &r=d.readers_[generation_&1].value;r.fetch_add(1,std::memory_order_seq_cst);
        if(d.generation_.load(std::memory_order_seq_cst)==generation_)break;
        r.fetch_sub(1,std::memory_order_seq_cst);
      }
    }
    ~Guard(){if(owner_)owner_->readers_[generation_&1].value.fetch_sub(1,std::memory_order_seq_cst);}
    Guard(Guard&&g)noexcept:owner_(std::exchange(g.owner_,nullptr)),generation_(g.generation_){}
    Guard(const Guard&)=delete;Guard&operator=(const Guard&)=delete;Guard&operator=(Guard&&)=delete;
  };
  Guard read(){return Guard(*this);}
  uint64_t generation()const{return generation_.load(std::memory_order_seq_cst);}
  void synchronize_detached(){
    std::lock_guard<std::mutex> lock(writer_);
    const auto old=generation_.fetch_add(1,std::memory_order_seq_cst);
    while(readers_[old&1].value.load(std::memory_order_seq_cst))
      std::this_thread::sleep_for(std::chrono::microseconds(20));
  }
};
}
