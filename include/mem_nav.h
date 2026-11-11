#pragma once
#include <atomic>
#include "utils/memnav_timing.h"
#include "utils/nav_read_domain.h"
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <stdexcept>
#include <exception>
#include <ostream>
namespace pipeann {
// Fixed storage, one graph writer. PQ rows immutable between release publication
// and its own old-reader grace period. Producers use a short lifecycle mutex; readers
// take no mutex. Capacity exhaustion is explicit, never silent sample loss.
class MemNav {
 public:
  static constexpr uint32_t invalid=UINT32_MAX;
  static constexpr uint64_t live_bit=uint64_t(1)<<63;
  using Pair=std::pair<float,uint32_t>;
  // Retain the guard until the caller finishes seed filtering, not just until
  // traversal returns. A slot handle/PQ pointer may not outlive this batch.
  struct CandidateBatch {
    NavReadDomain::Guard guard;
    std::vector<Pair> values;
    explicit CandidateBatch(NavReadDomain &d):guard(d){}
    CandidateBatch(CandidateBatch&&)=default;CandidateBatch(const CandidateBatch&)=delete;
    auto begin()const{return values.begin();}auto end()const{return values.end();}
    size_t size()const{return values.size();}bool empty()const{return values.empty();}
  };
  using Distance=std::function<float(const uint8_t*,const uint8_t*)>;
  struct Event {uint32_t id,gen;uint64_t tag;std::vector<uint8_t> pq;};
  const uint32_t max_ids,cap,R,chunks,build_L;
  MemNavTiming timing;
 private:
  std::unique_ptr<std::atomic<uint64_t>[]> identities_,id_state_;
  std::unique_ptr<std::atomic<uint32_t>[]> edges_,degrees_;
  std::unique_ptr<uint8_t[]> pq_;
  std::atomic<uint32_t> used_{0},ep_{invalid};
  std::vector<uint32_t> free_;
  std::unordered_map<uint64_t,uint32_t> tag_id_;
  std::unordered_map<uint32_t,uint32_t> id_slot_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Event> queue_;
  bool busy_=false,stop_=false;
  std::exception_ptr error_;
  std::thread writer_;
  Distance distance_;
  mutable NavReadDomain read_domain_;
  std::atomic<uint64_t> enqueued_{0},added_{0},stale_{0},removed_{0},reclaimed_{0},full_{0};
 public:
  static uint64_t pack(uint32_t id,uint32_t gen,bool live=true) {return uint64_t(id)|(uint64_t(gen)<<32)|(live?live_bit:0);}
  static uint32_t generation(uint64_t s) {return uint32_t(s>>32)&0x7fffffff;}
  static uint64_t hash(uint64_t x) {x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}
  // Deterministic Bernoulli-style hash sampling, reproducible across runs.
  static bool sample(uint64_t tag) {return hash(tag^0x20261111ULL)%100==0;}
  MemNav(uint32_t ids,uint32_t slots,uint32_t r,uint32_t c,Distance d,uint32_t l=96)
   :max_ids(ids),cap(slots),R(r),chunks(c),build_L(l),identities_(new std::atomic<uint64_t>[slots]),
    id_state_(new std::atomic<uint64_t>[ids]),edges_(new std::atomic<uint32_t>[size_t(slots)*r]),
    degrees_(new std::atomic<uint32_t>[slots]),pq_(new uint8_t[size_t(slots)*c]),distance_(std::move(d)) {
    if(!ids||!slots||!r||!c||l<r)throw std::invalid_argument("MemNav capacity");
    for(uint32_t i=0;i<ids;++i)id_state_[i].store(0);
    for(uint32_t s=0;s<slots;++s){identities_[s].store(0);degrees_[s].store(0);}
    for(size_t e=0;e<size_t(slots)*r;++e)edges_[e].store(invalid);
    writer_=std::thread([this]{run();});
  }
  ~MemNav() {{std::lock_guard<std::mutex> g(mu_);stop_=true;}cv_.notify_all();if(writer_.joinable())writer_.join();}
  auto read_guard()const{return read_domain_.read();}
  uint64_t reader_generation()const{return read_domain_.generation();}
  // A reader must retain CandidateBatch/read_guard while consuming this pointer.
  const uint8_t *code(uint32_t s) const {return pq_.get()+size_t(s)*chunks;}
  uint64_t identity(uint32_t s) const {return s<cap?identities_[s].load(std::memory_order_acquire):0;}
  bool valid_identity(uint64_t s) const {uint32_t id=uint32_t(s);return (s&live_bit)&&id<max_ids&&id_state_[id].load(std::memory_order_acquire)==s;}
  // After successful disk publication, still under insert/merge barrier. Caller
  // supplies owned PQ encoding; a delayed worker never reads a reused disk row.
  void insert(uint32_t id,uint64_t tag,std::vector<uint8_t> pq,bool live=true) {
    MemNavTiming::Scope timer(timing,MemNavTiming::InsertBookkeeping);
    if(id>=max_ids)throw std::overflow_error("MemNav ID capacity exhausted");
    std::lock_guard<std::mutex> g(mu_);if(error_)std::rethrow_exception(error_);
    uint32_t gen=generation(id_state_[id].load())+1;
    if(gen>=0x80000000U)throw std::overflow_error("MemNav generation exhausted");
    id_state_[id].store(pack(id,gen,live),std::memory_order_release);
    if(!live||!sample(tag))return;
    if(pq.size()!=chunks)throw std::invalid_argument("MemNav PQ length");
    if(tag_id_.count(tag))throw std::logic_error("duplicate sampled live tag");
    tag_id_[tag]=id;queue_.push_back({id,gen,tag,std::move(pq)});++enqueued_;cv_.notify_all();
  }
  void erase_tag(uint64_t tag) {
    MemNavTiming::Scope timer(timing,MemNavTiming::DeleteMark);
    std::lock_guard<std::mutex> g(mu_);auto it=tag_id_.find(tag);if(it==tag_id_.end())return;
    uint32_t id=it->second;auto old=id_state_[id].fetch_and(~live_bit,std::memory_order_acq_rel);
    auto si=id_slot_.find(id);
    if(si!=id_slot_.end()){uint64_t expected=old;if(identities_[si->second].compare_exchange_strong(expected,old&~live_bit))++removed_;}
    tag_id_.erase(it);
    auto ep=ep_.load(std::memory_order_acquire);
    if(ep!=invalid && !valid_identity(identity(ep))) {
      uint32_t replacement=invalid;
      for(uint32_t s=0;s<used_;++s)if(valid_identity(identity(s))){replacement=s;break;}
      ep_.store(replacement,std::memory_order_release);
    }
  }
  void flush() {
    std::unique_lock<std::mutex> g(mu_);cv_.wait(g,[&]{return error_||(queue_.empty()&&!busy_);});
    if(error_)std::rethrow_exception(error_);
  }
  template<class D> CandidateBatch candidates(D dist,uint32_t L) const {
    CandidateBatch batch(read_domain_);auto &result=batch.values;if(!L)return batch;
    uint32_t ep=ep_.load(std::memory_order_acquire);if(ep==invalid||!identity(ep))return batch;
    std::priority_queue<Pair,std::vector<Pair>,std::greater<Pair>> frontier;
    std::priority_queue<Pair> best;std::unordered_set<uint32_t> seen;
    auto add=[&](uint32_t s){
      if(s>=used_.load(std::memory_order_acquire)||!identity(s)||!seen.insert(s).second)return;
      float d=dist(code(s));if(best.size()<L||d<best.top().first){frontier.push({d,s});best.push({d,s});if(best.size()>L)best.pop();}
    };
    add(ep);
    while(!frontier.empty()){
      auto p=frontier.top();frontier.pop();if(best.size()==L&&p.first>best.top().first)break;
      uint32_t n=std::min(R,degrees_[p.second].load(std::memory_order_acquire));
      for(uint32_t i=0;i<n;++i)add(edges_[size_t(p.second)*R+i].load(std::memory_order_relaxed));
    }
    while(!best.empty()){result.push_back(best.top());best.pop();}std::reverse(result.begin(),result.end());return batch;
  }
  std::vector<uint32_t> prune(const uint8_t *center,std::vector<uint32_t> pool,uint32_t self) {
    std::sort(pool.begin(),pool.end());pool.erase(std::unique(pool.begin(),pool.end()),pool.end());
    std::vector<Pair> ranked;for(auto s:pool)if(s!=self&&s<used_&&valid_identity(identity(s)))ranked.push_back({distance_(center,code(s)),s});
    std::sort(ranked.begin(),ranked.end());std::vector<uint32_t> out;
    for(auto p:ranked){bool keep=true;for(auto q:out)if(1.2f*distance_(code(q),code(p.second))<p.first){keep=false;break;}
      if(keep){out.push_back(p.second);if(out.size()==R)break;}}
    return out;
  }
  void set_edges(uint32_t s,const std::vector<uint32_t>& v) {
    for(uint32_t i=0;i<R;++i)edges_[size_t(s)*R+i].store(i<v.size()?v[i]:invalid,std::memory_order_relaxed);
    degrees_[s].store(uint32_t(v.size()),std::memory_order_release);
  }
  std::vector<uint32_t> neighbors(uint32_t s) const {
    std::vector<uint32_t> v;for(uint32_t i=0;i<std::min(R,degrees_[s].load());++i){auto n=edges_[size_t(s)*R+i].load();if(n<cap)v.push_back(n);}return v;
  }
  void process(Event e) {
    MemNavTiming::Scope timer(timing,MemNavTiming::BackgroundInsert);
    auto want=pack(e.id,e.gen);if(id_state_[e.id].load(std::memory_order_acquire)!=want){++stale_;return;}
    auto found=candidates([&](const uint8_t *c){return distance_(e.pq.data(),c);},build_L);
    std::vector<uint32_t> pool;for(auto p:found)pool.push_back(p.second);
    auto links=prune(e.pq.data(),pool,invalid);uint32_t s;
    {std::lock_guard<std::mutex> g(mu_);if(id_state_[e.id].load()!=want){++stale_;return;}
      if(!free_.empty()){s=free_.back();free_.pop_back();}
      else{s=used_;if(s==cap){++full_;throw std::overflow_error("MemNav slot capacity exhausted");}used_.store(s+1);}
      std::copy(e.pq.begin(),e.pq.end(),pq_.get()+size_t(s)*chunks);set_edges(s,links);id_slot_[e.id]=s;
      identities_[s].store(want,std::memory_order_release);++added_;if(ep_.load()==invalid)ep_.store(s,std::memory_order_release);
    }
    for(auto n:links){auto v=neighbors(n);v.push_back(s);set_edges(n,prune(code(n),std::move(v),n));}
  }
  void run() {
    try{for(;;){Event e;
      {std::unique_lock<std::mutex> g(mu_);cv_.wait(g,[&]{return stop_||!queue_.empty();});if(stop_&&queue_.empty())break;
       e=std::move(queue_.front());queue_.pop_front();busy_=true;}
      process(std::move(e));{std::lock_guard<std::mutex> g(mu_);busy_=false;}cv_.notify_all();
    }}catch(...){std::lock_guard<std::mutex> g(mu_);error_=std::current_exception();busy_=false;cv_.notify_all();}
  }
  // Caller excludes producers and flushes the graph writer. Readers keep running.
  // First detach every dead slot from all live edges and the entry, publish those
  // stores with a generation rotation, then wait only for pre-rotation batches.
  // New batches acquire the new generation and cannot observe pre-detachment edges.
  // Never call this while holding a CandidateBatch/read_guard on this thread.
  void repair_and_reclaim() {
    MemNavTiming::Scope timer(timing,MemNavTiming::RepairTotal);
    std::lock_guard<std::mutex> g(mu_);if(busy_||!queue_.empty())throw std::logic_error("repair without flush");
    std::vector<uint32_t> dead;uint32_t first=invalid;
    { MemNavTiming::Scope graph_timer(timing,MemNavTiming::RepairGraph);
    for(uint32_t s=0;s<used_;++s)if(identity(s)){if(!valid_identity(identity(s)))dead.push_back(s);else if(first==invalid)first=s;}
    for(uint32_t s=0;s<used_;++s)if(valid_identity(identity(s))){
      auto old=neighbors(s),pool=old;bool changed=false;std::vector<uint32_t> todo;std::unordered_set<uint32_t> seen;
      for(auto n:old)if(!valid_identity(identity(n))){todo.push_back(n);changed=true;}
      for(size_t i=0;i<todo.size()&&i<build_L*4;++i){auto n=todo[i];if(!seen.insert(n).second)continue;
        for(auto q:neighbors(n)){if(valid_identity(identity(q)))pool.push_back(q);else if(identity(q))todo.push_back(q);}}
      if(changed){auto v=prune(code(s),std::move(pool),s);if(v.empty()&&first!=s&&first!=invalid)v.push_back(first);set_edges(s,v);}
    }
    ep_.store(first,std::memory_order_release);
    }
    // With no retired slots there is nothing whose payload could be overwritten.
    if(dead.empty())return;
    {MemNavTiming::Scope grace_timer(timing,MemNavTiming::NavGraceWait);
     read_domain_.synchronize_detached();}
    MemNavTiming::Scope reclaim_timer(timing,MemNavTiming::Reclaim);
    for(auto s:dead){auto ident=identity(s);auto it=id_slot_.find(uint32_t(ident));if(it!=id_slot_.end()&&it->second==s)id_slot_.erase(it);
      identities_[s].store(0,std::memory_order_release);degrees_[s].store(0);free_.push_back(s);++reclaimed_;}
  }
  void audit(std::ostream &o) const {
    uint64_t live=0,dead=0,bad=0,degree_sum=0;uint32_t degree_min=R,degree_max=0;std::unordered_set<uint32_t> seen;
    for(uint32_t s=0;s<used_;++s)if(identity(s)){if(valid_identity(identity(s))){++live;auto deg=degrees_[s].load();degree_sum+=deg;degree_min=std::min(degree_min,deg);degree_max=std::max(degree_max,deg);}else ++dead;for(auto n:neighbors(s))if(!identity(n))++bad;}
    std::vector<uint32_t> todo;auto ep=ep_.load();if(ep!=invalid)todo.push_back(ep);
    for(size_t i=0;i<todo.size();++i){auto s=todo[i];if(!seen.insert(s).second)continue;for(auto n:neighbors(s))if(identity(n))todo.push_back(n);}
    o<<"{\"live\":"<<live<<",\"dead\":"<<dead<<",\"reachable\":"<<seen.size()<<",\"invalid_edges\":"<<bad
     <<",\"enqueued\":"<<enqueued_<<",\"added\":"<<added_<<",\"stale_events\":"<<stale_<<",\"removed\":"<<removed_
     <<",\"reclaimed\":"<<reclaimed_<<",\"capacity_errors\":"<<full_
     <<",\"degree_min\":"<<(live?degree_min:0)<<",\"degree_max\":"<<degree_max<<",\"degree_mean\":"<<(live?double(degree_sum)/live:0)<<"}";
    if(bad)throw std::runtime_error("MemNav invalid edges");
  }
};
}
