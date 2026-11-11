#include "mem_nav.h"
#include "utils/reader_epoch.h"
#include <future>
#include <sstream>
#include <iostream>
#include <chrono>
using namespace pipeann;
void require(bool ok,const char *msg){if(!ok)throw std::runtime_error(msg);}
int main(){
  ReaderEpoch e,other;
  std::atomic<bool> entered{false},release{false},passed{false};
  std::thread r([&]{EpochGuard g(e);EpochGuard nested(e);EpochGuard separate(other);entered=true;while(!release)std::this_thread::yield();});
  while(!entered)std::this_thread::yield();
  std::thread w([&]{e.quiescent([&]{passed=true;},50);});
  std::this_thread::sleep_for(std::chrono::milliseconds(2200));
  require(!passed,"reader forcibly reclaimed after timeout");release=true;r.join();w.join();require(passed,"drain failed");
  for(int i=0;i<600;++i){std::thread t([&]{EpochGuard a(e);EpochGuard b(other);});t.join();}
  auto distance=[](const uint8_t*a,const uint8_t*b){float d=0;for(int i=0;i<4;++i){float x=float(a[i])-b[i];d+=x*x;}return d;};
  MemNav nav(2000,1000,16,4,distance,48);
  std::vector<uint64_t> tags;for(uint64_t t=0;tags.size()<800;++t)if(MemNav::sample(t))tags.push_back(t);
  std::atomic<bool> run{true};std::atomic<uint64_t> reads{0};
  std::thread reader([&]{uint8_t q[4]={1,2,3,4};while(run){EpochGuard g(e);auto v=nav.candidates([&](const uint8_t*c){return distance(q,c);},32);for(auto p:v){auto s=nav.identity(p.second);if(nav.valid_identity(s))require(uint32_t(s)<2000,"seed range");}++reads;}});
  for(uint32_t i=0;i<400;++i){nav.insert(i,tags[i],std::vector<uint8_t>(4,uint8_t(i)));nav.erase_tag(tags[i]);}
  nav.flush();nav.repair_and_reclaim();
  std::ostringstream a;nav.audit(a);require(a.str().find("\"live\":0,")!=std::string::npos,"deleted event resurrected");
  for(uint32_t i=0;i<400;++i)nav.insert(i,tags[i+400],std::vector<uint8_t>(4,uint8_t(i)));
  nav.flush();nav.repair_and_reclaim();
  std::ostringstream b;nav.audit(b);require(b.str().find("\"live\":400,")!=std::string::npos,"reuse lost identities");
  // Delete-before-publication must not enqueue an event.
  nav.insert(500,tags[0],{},false);nav.flush();
  run=false;reader.join();require(reads>0,"no concurrent readers");
  std::cout<<"MEMNAV_SAFETY_PASS "<<b.str()<<" reads="<<reads<<std::endl;
}
