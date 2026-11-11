#pragma once
#include <atomic>
#include <array>
#include <chrono>
#include <ctime>
#include <ostream>
namespace pipeann {
struct MemNavTiming {
  enum Key {Admission, InsertBookkeeping, DeleteMark, BackgroundInsert, MergeTotal,
            MergeRemark, MergeFlush, EpochCall, EpochWait, RepairTotal, RepairGraph, Reclaim, NavGraceWait, Count};
  static uint64_t wall(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
  static uint64_t cpu(){timespec t{};clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return uint64_t(t.tv_sec)*1000000000ULL+t.tv_nsec;}
  struct Counter {std::atomic<uint64_t> calls{0},wall_ns{0},cpu_ns{0},max_wall_ns{0};};
  std::array<Counter,Count> counters;
  void add(Key k,uint64_t w,uint64_t c){auto &v=counters[k];v.calls.fetch_add(1,std::memory_order_relaxed);v.wall_ns.fetch_add(w,std::memory_order_relaxed);v.cpu_ns.fetch_add(c,std::memory_order_relaxed);auto m=v.max_wall_ns.load(std::memory_order_relaxed);while(w>m&&!v.max_wall_ns.compare_exchange_weak(m,w,std::memory_order_relaxed)){} }
  struct Scope {MemNavTiming &t;Key k;uint64_t w,c;Scope(MemNavTiming &t,Key k):t(t),k(k),w(wall()),c(cpu()){}~Scope(){auto endcpu=cpu();auto endwall=wall();t.add(k,endwall-w,endcpu-c);}Scope(const Scope&)=delete;};
  // Only reset at a round boundary with producers idle and the graph writer flushed.
  void reset(){for(auto &v:counters){v.calls=0;v.wall_ns=0;v.cpu_ns=0;v.max_wall_ns=0;}}
  void dump(std::ostream&o)const {
    const char *names[]={"foreground_admission","insert_bookkeeping","delete_mark","background_graph_insert","merge_maintenance_total","merge_repeat_delete","merge_queue_drain","epoch_call_total","epoch_admission_and_reader_drain","repair_total","repair_scan_bridge_prune","slot_reclaim","nav_old_reader_grace_wait"};
    o<<"{";for(size_t i=0;i<Count;++i){auto &v=counters[i];if(i)o<<",";o<<"\""<<names[i]<<"\":{\"calls\":"<<v.calls.load()<<",\"wall_ns\":"<<v.wall_ns.load()<<",\"thread_cpu_ns\":"<<v.cpu_ns.load()<<",\"max_wall_ns\":"<<v.max_wall_ns.load()<<"}";}o<<"}";
  }
};
}
