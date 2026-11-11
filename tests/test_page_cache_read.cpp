#include "utils/page_cache.h"
#include <array>
#include <thread>
#include <atomic>
#include <stdexcept>
#include <iostream>
void require(bool b){if(!b)throw std::runtime_error("late read write-cache regression");}
int main(){
  std::array<uint8_t,SECTOR_LEN> fresh,old,observed;
  fresh.fill(77);old.fill(11);
  // Reproduce old API behavior: late read poisoned an already-published image.
  auto &bad=pipeann::cache;bad.put(1,fresh.data(),true);bad.put(1,old.data(),true);
  require(bad.get(1,observed.data())&&observed[0]==11);bad.deref(1);bad.deref(1);
  // Actual fixed completion method: cover existing and evicted write images.
  auto &c=pipeann::cache;
  for(int k=0;k<1000;++k){
    std::atomic<bool> staged{false};std::vector<uint64_t> refs;old.fill(11);
    std::thread writer([&]{c.put(7,fresh.data(),true);staged.store(true,std::memory_order_release);});
    while(!staged.load(std::memory_order_acquire))std::this_thread::yield();
    c.finish_read(7,old.data(),&refs);writer.join();
    require(refs.size()==1&&old[0]==77&&old.back()==77);
    require(c.get(7,observed.data())&&observed==fresh);
    c.deref(7);c.deref(refs[0]);require(!c.get(7,observed.data()));
    old.fill(11);refs.clear();c.finish_read(7,old.data(),&refs);
    require(refs.empty()&&old[0]==11&&!c.get(7,observed.data()));
  }
  // A multi-sector read with one hit must not acquire phantom miss references.
  c.put(10,fresh.data(),true);std::vector<uint64_t> refs;
  c.finish_read(10,old.data(),&refs);c.finish_read(11,old.data(),&refs);
  require(refs.size()==1&&refs[0]==10);c.deref(10);c.deref(10);
  require(c.cache.size()==0);
  std::cout<<"PAGE_CACHE_LATE_READ_REGRESSION_PASS\n";
}
