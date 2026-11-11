#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace pipeann {
  struct QueryStats {
    double total_us = 0;        // total time to process query in micros
    double n_4k = 0;            // # of 4kB reads
    double n_8k = 0;            // # of 8kB reads
    double n_12k = 0;           // # of 12kB reads
    double n_ios = 0;           // total # of IOs issued
    double read_size = 0;       // total # of bytes read
    double io_us = 0;           // total time spent in IO
    double io_us1 = 0;          // total time spent in IO
    double head_us = 0;         // total time spent in in-memory index
    double cpu_us = 0;          // total time spent in CPU
    double cpu_us1 = 0;         // total time spent in CPU
    double cpu_us2 = 0;         // total time spent in CPU
    double n_cmps_saved = 0;    // # cmps saved
    double n_cmps = 0;          // # cmps
    double n_cache_hits = 0;    // # cache_hits
    double n_hops = 0;          // # search hops
    double n_current_used = 0;  // # force return for latency limit
    double rerank_us = 0;       // Phase 2 rerank time (exact vector reads + distance compute)
  };

  inline double get_percentile_stats(QueryStats *stats, uint64_t len, float percentile,
                                     const std::function<double(const QueryStats &)> &member_fn) {
    std::vector<double> vals(len);
    for (uint64_t i = 0; i < len; i++) {
      vals[i] = member_fn(stats[i]);
    }

    std::sort(vals.begin(), vals.end(), [](const double &left, const double &right) { return left < right; });

    auto retval = vals[(uint64_t) (percentile * ((float) len))];
    vals.clear();
    return retval;
  }

  inline double get_mean_stats(QueryStats *stats, uint64_t len,
                               const std::function<double(const QueryStats &)> &member_fn) {
    double avg = 0;
    for (uint64_t i = 0; i < len; i++) {
      avg += member_fn(stats[i]);
    }
    return avg / ((double) len);
  }

  // InsertStats timing model (decouple_insert_in_place):
  //
  //   total_us  ≈  search_us + vec_write_us + graph_commit_us + other_us
  //
  //   graph_commit_us (= update_topo_us) is the sum of mutually exclusive buckets:
  //     find_graph_loc_us   neighbor-page lock + alloc_graph_loc (+ retries)
  //     lock_wait_us        extra lock on newly allocated graph pages only
  //     read_topo_us        graph read_alloc only
  //     prune_us              neighbor reverse-edge loop (incl. delta_prune); printed as graph_commit "prune"
  //     build_topo_us           full in-memory build incl. split copy + target; used in update_topo_us sum
  //     ensure_vec_us       retired (always 0): visibility now comes from the in-memory
  //                         vector overlay consulted by search, not a synchronous flush
  //     publish_cache_us    set_id2loc + wbc_write + split id2loc flip
  //     write_topo_us       graph_reader_->write (persist graph to disk)
  //
  //   prune_us is a detail metric only: time inside delta_prune_neighbors, already
  //   included in build_topo_us. Do NOT add prune_us into graph_commit_us.
  //
  //   other_us (not stored) covers nbr_handler->insert, prune_neighbors, buffer
  //   setup, unlock/deref, push_query_buf, etc.
  struct InsertStats {
    double total_us = 0;

    double search_us = 0;           // internal decouple search
    double search_rerank_us = 0;    // exact-vector rerank within decouple search
    double vec_write_us = 0;        // stage_decouple_vector_write at insert start

    double find_graph_loc_us = 0;   // graph_commit: neighbor lock + alloc
    double lock_wait_us = 0;        // graph_commit: lock new/tail graph pages
    double read_topo_us = 0;        // graph_commit: graph read_alloc
    double build_topo_us = 0;       // graph_commit: in-memory topology build
    double ensure_vec_us = 0;       // graph_commit: pwrite target vector before publish (keeps sector buffer)
    double publish_cache_us = 0;    // graph_commit: id2loc + wbc_write (+ split idx)
    double write_topo_us = 0;       // graph_commit: graph disk write

    double prune_us = 0;            // detail: delta_prune inside build_topo_us

    // find_graph_loc_us breakdown (detail, all subsets of find_graph_loc_us):
    double find_loc_lock_us = 0;        // cumulative neighbor-page lockReqs (incl. retries)
    double find_loc_alloc_us = 0;       // alloc_graph_loc total (alloc_lock wait + work)
    double find_loc_alloc_wait_us = 0;  // time blocked acquiring the global alloc_lock
    double find_loc_retries = 0;        // # of retry iterations (neighbor relocated mid-lock)

    // Convenience sum of graph_commit buckets above (excludes search/vec/prune_us).
    double update_topo_us = 0;
  };
}  // namespace pipeann
