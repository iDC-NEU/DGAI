#include "aligned_file_reader.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#ifndef USE_AIO
#include "liburing.h"
#endif

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include "utils/timer.h"
#include "utils/tsl/robin_set.h"
#include "utils.h"
#include "utils/page_cache.h"
#include "linux_aligned_file_reader.h"

#include <unistd.h>
#include <sys/syscall.h>
#include <mutex>
#include <sstream>
#include <queue>
#include <unordered_map>
#include <deque>

namespace pipeann {

// Minimal atomic logger for diagnosing idx lock contention.
inline void search_log(const std::string &msg) {
  // static std::mutex mu;
  // auto tid = syscall(SYS_gettid);
  // auto ts = std::chrono::steady_clock::now().time_since_epoch().count() / 1000000;
  // std::lock_guard<std::mutex> lk(mu);
  // std::cerr << "[tid=" << tid << " ts=" << ts << "] SEARCH: " << msg << std::endl;
}
  // Per-page I/O tracking for decoupled search.
  // When multiple nodes reside on the same page, only one I/O is issued;
  // subsequent nodes reuse the same buffer pointer.
  struct PageIOInfo {
    IORequest *req;
    unsigned rk;        // rank (position in retset) when the first I/O for this page was issued
    uint32_t base_loc = 0;  // first loc on this page (pid * nnodes_per_sector_graph)
    // loc2id snapshot at read time, indexed by (loc - base_loc). Used to detect
    // a stale snapshot: if a concurrent split relocated a node onto this page
    // after we read it, the node deduping here did NOT own its loc at read time,
    // so the buffer holds someone else's bytes and we must re-read.
    std::vector<uint32_t> slot_owners{};
    uint32_t seq_v1 = 0;  // page seqlock sample taken before issuing the read
  };

  // A graph node whose I/O has completed and is waiting to be expanded.
  // Decouples I/O completion (poll_all) from CPU expansion (expand_one).
  struct ReadyEntry {
    unsigned id;
    float distance;
  };
#define CACHE_N
#define USE_PRIORITY_QUEUE
#ifdef USE_PRIORITY_QUEUE
  // I/O tracking struct for decoupled search (graph-only reads).
  struct decouple_io_t {
    Neighbor nbr;
    unsigned page_id;
    unsigned loc;
    IORequest *read_req;
    unsigned rk;         // rank when this I/O was logically issued (inherited from page's first request)
    uint64_t timestamp;  // global timestamp for FIFO tie-breaking among same-page requests
    bool finished() const {
      return read_req->finished;
    }
  };

  // Priority comparator for the min-heap.
  // Priority order (higher priority = popped first):
  //   1. completed I/Os (flag 0) before in-flight I/Os (flag 1)
  //   2. lower rk before higher rk
  //   3. lower timestamp before higher timestamp (FIFO for same-page requests)
  struct DecoupleIOCompare {
    bool operator()(const decouple_io_t &a, const decouple_io_t &b) const {
      int a_flag = !a.finished();
      int b_flag = !b.finished();
      if (a_flag != b_flag) return a_flag > b_flag;
      if (a.rk != b.rk) return a.rk > b.rk;
      return a.timestamp > b.timestamp;
    }
  };
#else
  // I/O tracking struct for decoupled search (graph-only reads).
  struct decouple_io_t {
    Neighbor nbr;
    unsigned page_id;
    unsigned loc;
    IORequest *read_req;
    bool operator>(const decouple_io_t &rhs) const {
      return nbr.distance > rhs.nbr.distance;
    }
    bool operator<(const decouple_io_t &rhs) const {
      return nbr.distance < rhs.nbr.distance;
    }
    bool finished() const {
      return read_req->finished;
    }
  };
#endif

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::do_decouple_search(const T *query, uint32_t mem_L, uint32_t Lsize,
                                              const uint32_t beam_width,
                                              std::vector<Neighbor> &expanded_nodes_info,
                                              InsertContext *insert_ctx, QueryStats *stats) {
    decouple_search(query, Lsize, mem_L, Lsize, expanded_nodes_info,
                    beam_width, stats,
                    nullptr, nullptr,
                    Lsize, insert_ctx);
  }

  // Synchronous decoupled beam search – identical structure to do_beam_search.
  // Differs only in: graph_reader_ (no coords in page), Phase 2 exact-vector reads.
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::do_decouple_beam_search(
      const T *query1, uint32_t mem_L, uint32_t l_search, const uint32_t beam_width,
      std::vector<Neighbor> &expanded_nodes_info,
      tsl::robin_map<uint32_t, T *> *coord_map, T *coord_buf, QueryStats *stats,
      tsl::robin_set<uint32_t> *exclude_nodes, bool dyn_search_l,
      std::vector<uint64_t> *passthrough_page_ref) {
#ifdef DGAI_MERGE_COORD_EXCLUSIVE
    std::shared_lock<std::shared_mutex> lk(merge_lock);
#else
    pipeann::EpochGuard epoch_guard(reader_epoch_);
#endif
    uint32_t original_l_search = l_search;
    auto diskSearchBegin = std::chrono::high_resolution_clock::now();

    auto query_buf = pop_query_buf(query1);
    void *ctx = graph_reader_->get_ctx();

    const T *query = query_buf->aligned_query_T;

    // reset query
    query_buf->reset();

    // pointers to current vector for comparison
    T *data_buf = query_buf->coord_scratch;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_buf->sector_scratch;
    uint64_t &sector_scratch_idx = query_buf->sector_idx;

    nbr_handler->initialize_query(query, query_buf);
    float *dist_scratch = query_buf->aligned_dist_scratch;

    Timer query_timer, io_timer, cpu_timer;
    std::vector<Neighbor> retset;
    retset.resize(mem_L + 10 * l_search);
    tsl::robin_set<uint64_t> visited(4096);

    // re-naming `expanded_nodes_info` to not change rest of the code
    std::vector<Neighbor> &full_retset = expanded_nodes_info;
    full_retset.reserve(10 * l_search);

    unsigned cur_list_size = 0;
    auto compute_and_add_to_retset = [&](const unsigned *node_ids, const uint64_t n_ids) {
      nbr_handler->compute_dists(query_buf, node_ids, n_ids);
      for (uint64_t i = 0; i < n_ids; ++i) {
#ifdef DGAI_STRICT_GRAPH_CHECKS
        check_graph_node_mapping(node_ids[i], id2loc(node_ids[i]), "decouple_beam_search/seed");
#endif
        retset[cur_list_size].id = node_ids[i];
        retset[cur_list_size].distance = dist_scratch[i];
        retset[cur_list_size++].flag = true;
        visited.insert(node_ids[i]);
      }
    };

    if (mem_L) {
      // Both MemNav and legacy mem_topo return live internal IDs through this API.
      std::vector<uint32_t> live_ids(mem_L);
      const uint32_t n_mem =
          mem_search_ids_with_pq(query, query_buf, mem_L, live_ids.data(), dist_scratch);
      const uint32_t n_live = n_mem;
      uint32_t seeded = 0;
      for (uint64_t i = 0; i < std::min((uint64_t) n_live, (uint64_t) l_search); ++i) {
        retset[cur_list_size].id = live_ids[i];
        retset[cur_list_size].distance = dist_scratch[i];
        retset[cur_list_size++].flag = true;
        visited.insert(live_ids[i]);
        ++seeded;
      }
      if (seeded == 0) {
        compute_and_add_to_retset(&meta_.entry_point_id, 1);
      }
    } else {
      // Do not use optimized start point.
      compute_and_add_to_retset(&meta_.entry_point_id, 1);
    }

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned cmps = 0;
    unsigned hops = 0;
    unsigned num_ios = 0;
    unsigned k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    using fnhood_t = std::tuple<unsigned, unsigned, char *>;
    std::vector<fnhood_t> frontier_nhoods;
    std::vector<IORequest> frontier_read_reqs;
    std::vector<uint32_t> vec_rdlocks;

    std::vector<uint64_t> new_page_ref{};
    std::vector<uint64_t> &page_ref = passthrough_page_ref ? *passthrough_page_ref : new_page_ref;

    uint64_t coord_buf_idx = 0;  // used by coord_map
    while (k < cur_list_size) {
      auto nk = cur_list_size;
      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      vec_rdlocks.clear();
      sector_scratch_idx = 0;
      // find new beam
      // WAS: uint64_t marker = k - 1;
      uint32_t marker = k;
      uint32_t num_seen = 0;
      while (marker < cur_list_size && frontier.size() < beam_width && num_seen < beam_width) {
        if (retset[marker].flag) {
          num_seen++;
          frontier.push_back(retset[marker].id);
          retset[marker].flag = false;
        }
        marker++;
      }

      // read nhoods of frontier ids
      std::vector<uint32_t> locked;
      if (!frontier.empty()) {
        if (stats != nullptr)
          stats->n_hops++;
        // {
        //   std::ostringstream oss;
        //   oss << "idx RD LOCK frontier n=" << frontier.size();
        //   if (frontier.size() > 0) {
        //     oss << " sample=[";
        //     for (size_t si = 0; si < std::min(frontier.size(), size_t(8)); ++si)
        //       oss << (si ? " " : "") << frontier[si];
        //     oss << "]";
        //   }
        //   search_log(oss.str());
        // }
        locked = this->lock_idx(idx_lock_table, kInvalidID, frontier, true);
        for (uint64_t i = 0; i < frontier.size(); i++) {
          uint32_t id = frontier[i];
          uint32_t loc = this->id2loc(id);
#ifdef DGAI_STRICT_GRAPH_CHECKS
          check_graph_node_mapping(id, loc, "decouple_beam_search/frontier");
#endif
          uint64_t offset = graph_loc_sector_no(loc) * SECTOR_LEN;
          auto sector_buf = sector_scratch + sector_scratch_idx * size_per_io_graph_;
          fnhood_t fnhood = std::make_tuple(id, loc, sector_buf);
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          frontier_read_reqs.emplace_back(
              IORequest(offset, size_per_io_graph_, sector_buf, 0, 0));
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
          num_ios++;
        }
        io_timer.reset();
#ifndef DGAI_MERGE_COORD_EXCLUSIVE
        // Seqlock: sample versions before the read; retry any torn page.
        std::vector<uint32_t> frontier_seq_v1(frontier_read_reqs.size());
        for (size_t i = 0; i < frontier_read_reqs.size(); ++i) {
          const uint64_t pid = frontier_read_reqs[i].offset / SECTOR_LEN;
          frontier_seq_v1[i] = page_ver_load(pid);
        }
#endif
        graph_reader_->read_alloc(frontier_read_reqs, ctx, &page_ref);
#ifndef DGAI_MERGE_COORD_EXCLUSIVE
        for (size_t i = 0; i < frontier_read_reqs.size(); ++i) {
          const uint64_t pid = frontier_read_reqs[i].offset / SECTOR_LEN;
          for (int attempt = 0; attempt < 1024; ++attempt) {
            std::atomic_thread_fence(std::memory_order_acquire);
            const uint32_t v1 = frontier_seq_v1[i];
            const uint32_t v2 = page_ver_load(pid);
            if (!(v1 & 1) && v1 == v2) break;
            frontier_seq_v1[i] = page_ver_load(pid);
            if (frontier_seq_v1[i] & 1) {
              _mm_pause();
              continue;
            }
            std::vector<IORequest> retry{frontier_read_reqs[i]};
            graph_reader_->read_alloc(retry, ctx, &page_ref);
            frontier_read_reqs[i] = retry[0];
          }
        }
#endif

        if (stats != nullptr) {
          stats->io_us += (double) io_timer.elapsed();
        }
        this->unlock_idx(idx_lock_table, locked);
        search_log("idx RD UNLOCK frontier n=" + std::to_string(locked.size()));
      }

      for (auto &frontier_nhood : frontier_nhoods) {
        auto [id, loc, sector_buf] = frontier_nhood;
        DiskGraphNode<T> node = graph_node_from_page(sector_buf, loc, id, "decouple_beam_search/expand");

        // Graph node has no coords — record with approximate (PQ) distance.
        // Phase 2 will overwrite with exact distance.
        float cur_expanded_dist = retset[k].distance;
        full_retset.push_back(Neighbor(id, cur_expanded_dist, true));

        // compute node_nbrs <-> query dist in PQ space
        cpu_timer.reset();
        nbr_handler->compute_dists(query_buf, node.nbrs, node.nnbrs);
        if (stats != nullptr) {
          stats->n_cmps += (double) node.nnbrs;
          stats->cpu_us += (double) cpu_timer.elapsed();
        }

        cpu_timer.reset();
        // process prefetch-ed nhood
        for (uint64_t m = 0; m < node.nnbrs; ++m) {
          unsigned id = node.nbrs[m];
          if (visited.find(id) != visited.end()) {
            continue;
          }
          visited.insert(id);
#ifdef DGAI_STRICT_GRAPH_CHECKS
          check_graph_node_mapping(id, id2loc(id), "decouple_beam_search/expand_nbr");
#endif
          cmps++;
          float dist = dist_scratch[m];
          if (stats != nullptr) {
            stats->n_cmps++;
          }
          if (dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search))
            continue;
          Neighbor nn(id, dist, true);
          // variable search_L for deleted nodes.
          // Return position in sorted list where nn inserted.

          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);

          if (cur_list_size < l_search) {
            ++cur_list_size;
            if (unlikely(cur_list_size >= retset.size())) {
              retset.resize(2 * cur_list_size);
            }
          }

          if (r < nk)
            nk = r;  // nk logs the best position in the retset that was
                     // updated due to neighbors of n.
        }

        if (dyn_search_l && exclude_nodes != nullptr) {
          // TODO(gh): contention still exists in id2tag(x)
          // O(n), but it is not slow as L is typically smaller than 300.
          // l_search monotonically increases to handle deleted nodes.
          uint32_t tot = 0, cur = 0;
          for (cur = 0; cur < cur_list_size; ++cur) {
            uint32_t tag = id2tag(retset[cur].id);
            if (exclude_nodes->find(tag) == exclude_nodes->end()) {
              ++tot;
              if (tot == original_l_search) {
                break;
              }
            }
          }
          // cur is the stopped index (cur + 1 is the length it should be)
          l_search = std::max(original_l_search, cur + 1);
        }

        if (stats != nullptr) {
          stats->cpu_us += (double) cpu_timer.elapsed();
        }
      }

      // update best inserted position
      //

      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;

      hops++;
      if (stats != nullptr && stats->n_current_used != 0) {
        auto diskSearchEnd = std::chrono::high_resolution_clock::now();
        double elapsedSeconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(diskSearchEnd - diskSearchBegin).count();
        if (elapsedSeconds >= stats->n_current_used)
          break;
      }
    }

    // --- Phase 2: read exact vectors and recompute distances ---
    // Graph nodes lack coords; we re-read from vector file for final ranking.
    void *vec_ctx = vector_reader_->get_ctx();
    const uint64_t vec_data_len = meta_.data_dim * sizeof(T);
    for (size_t batch_start = 0; batch_start < full_retset.size(); batch_start += MAX_IO_BATCH) {
      size_t batch_end = std::min(batch_start + MAX_IO_BATCH, full_retset.size());
      size_t batch_size = batch_end - batch_start;

      // Build the disk-read batch, but first resolve any vector still staged in the
      // in-memory overlay (freshly inserted, not yet flushed). Overlay hits are copied
      // straight into the slot buffer and excluded from the disk read; overlay misses
      // are guaranteed to be on disk (the combiner only erases after the disk write).
      std::vector<IORequest> batch;
      batch.reserve(batch_size);
      for (size_t j = batch_start; j < batch_end; ++j) {
        uint32_t bid = full_retset[j].id;
        uint64_t u_off = vec_sector_offset(bid);
        size_t slot = j - batch_start;
        char *bbuf = sector_scratch + slot * size_per_io_vec_;
        if (try_read_decouple_vector(bid, bbuf + u_off)) {
          continue;
        }
        batch.push_back(IORequest(vec_loc_sector_no(bid) * SECTOR_LEN, size_per_io_vec_, bbuf,
                                  u_off, vec_data_len, sector_scratch));
      }
      if (!batch.empty()) {
        io_timer.reset();
        vector_reader_->read(batch, vec_ctx);
        if (stats != nullptr) {
          stats->io_us += (double) io_timer.elapsed();
        }
      }

      for (size_t i = 0; i < batch_size; ++i) {
        uint32_t id = full_retset[batch_start + i].id;
        uint64_t u_off = vec_sector_offset(id);
        char *buf = sector_scratch + i * size_per_io_vec_;
        const T *raw_vec = (const T *)(buf + u_off);
        float exact_dist = dist_cmp->compare(query, raw_vec, (unsigned)aligned_dim);
        full_retset[batch_start + i].distance = exact_dist;

        // Populate coord_map (same as do_beam_search).
        if (coord_map != nullptr) {
          const uint64_t coord_cap = static_cast<uint64_t>(this->params.L) * INSERT_COORD_BUF_L_FACTOR;
          if (unlikely(coord_buf == nullptr || coord_buf_idx >= coord_cap)) {
            LOG(ERROR) << "Please allocate larger coord_buf.";
            crash();
          }
          T *coord_ptr = coord_buf + coord_buf_idx * aligned_dim;
          memcpy(coord_ptr, raw_vec, vec_data_len);
          coord_map->insert(std::make_pair(id, coord_ptr));
          coord_buf_idx++;
        }
      }
    }

    // re-sort by distance
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    if (passthrough_page_ref == nullptr) {
      graph_reader_->deref(&page_ref, ctx);
    }

    push_query_buf(query_buf);

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::decouple_search(const T *query1, const uint64_t k_search, const uint32_t mem_L,
                                            const uint64_t active_l_search, std::vector<Neighbor> &expanded_nodes_info,
                                            const uint64_t beam_width, QueryStats *stats,
                                            TagT *res_tags, float *distances,
                                            uint64_t rerank_window,
                                            InsertContext *insert_ctx) {
    if (!decouple_mode_) {
      LOG(ERROR) << "decouple_search called but index is not in decouple mode";
      crash();
    }

#ifdef DGAI_MERGE_COORD_EXCLUSIVE
    std::shared_lock<std::shared_mutex> lk(merge_lock);
#else
    pipeann::EpochGuard epoch_guard(reader_epoch_);
#endif

    // The routing-only entry must not consume one of the user's candidate
    // slots, including when search L equals K. Insertion keeps its old budget.
    const uint64_t l_search = active_l_search +
        (insert_ctx == nullptr && is_routing_only(frozen_entry_id_) ? 1 : 0);
    QueryBuffer<T> *query_buf = pop_query_buf(query1);
#ifdef USE_AIO
    void *ctx = graph_reader_->get_ctx();
#else
    void *ctx = graph_reader_->get_ctx(IORING_SETUP_SQPOLL);  // use SQ polling only for pipe search.
#endif

    if (beam_width > MAX_N_SECTOR_READS) {
      LOG(ERROR) << "Beamwidth can not be higher than MAX_N_SECTOR_READS";
      crash();
    }

    const T *query = query_buf->aligned_query_T;
    query_buf->reset();

    T *data_buf = query_buf->coord_scratch;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    char *sector_scratch = query_buf->sector_scratch;
    float *dist_scratch = query_buf->aligned_dist_scratch;

    Timer query_timer;
#ifdef CACHE_N
    const uint64_t extendL = l_search * 2;
#else
    const uint64_t extendL = l_search;
#endif
    std::vector<Neighbor> retset(mem_L + this->params.R + extendL * 10);
    auto &visited = *(query_buf->visited);
    unsigned cur_list_size = 0;

    // full_retset stores candidates with approximate distances during traversal.
    // After traversal, we recompute exact distances.
    std::vector<Neighbor> full_retset;
    full_retset.reserve(l_search * 10);

    // We don't have exact coords during traversal, so we record visited node IDs
    // and later compute exact distances by reading the vector file.
    // For now, we accumulate node IDs into full_retset with their approximate distances.

#ifndef OVERLAP_INIT
    nbr_handler->initialize_query(query, query_buf);
#endif

    // Note: we cannot compute exact distances during traversal since we don't read vectors.
    // Instead, we just record the node ID with its approximate distance.
    auto record_visited_node = [&](const unsigned id, const float approx_dist) {
      full_retset.push_back(Neighbor(id, approx_dist, true));
    };

    uint64_t n_computes = 0;
    std::vector<unsigned> nbr_filtered;
    auto compute_and_push_nbrs = [&](DiskGraphNode<T> &node, uint32_t node_id, unsigned &nk) {
#ifdef DGAI_STRICT_GRAPH_CHECKS
      if (unlikely(node.nnbrs > meta_.range)) {
        check_graph_nnbrs(node.nnbrs, node_id, id2loc(node_id), "decouple_search/expand");
      }
#endif
      unsigned nbors_cand_size = 0;
      nbr_filtered.resize(node.nnbrs);
      for (unsigned m = 0; m < node.nnbrs; ++m) {
        const uint32_t nbr_id = node.nbrs[m];
#ifdef DGAI_STRICT_GRAPH_CHECKS
        const uint32_t checked_nbr_loc = id2loc(nbr_id);
        if (unlikely(checked_nbr_loc == kInvalidID)) {
          LOG(ERROR) << "Invalid expanded edge source_id=" << node_id
                     << " source_loc=" << id2loc(node_id) << " nbr_id=" << nbr_id;
        }
        check_graph_node_mapping(nbr_id, checked_nbr_loc, "decouple_search/expand_nbr");
#endif
        if (visited.find(nbr_id) == visited.end()) {
          nbr_filtered[nbors_cand_size++] = nbr_id;
          visited.insert(nbr_id);
        }
      }

      n_computes += nbors_cand_size;
      if (nbors_cand_size) {
        nbr_handler->compute_dists(query_buf, nbr_filtered.data(), nbors_cand_size);
        for (unsigned m = 0; m < nbors_cand_size; ++m) {
          const int nbor_id = nbr_filtered[m];
          const float nbor_dist = dist_scratch[m];
          if (stats != nullptr) {
            stats->n_cmps++;
          }

          if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == extendL))
            continue;
          Neighbor nn(nbor_id, nbor_dist, true);
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
          if (cur_list_size < extendL) {
            ++cur_list_size;
            if (unlikely(cur_list_size >= retset.size())) {
              retset.resize(2 * cur_list_size);
            }
          }
          if (r < nk)
            nk = r;
        }
      }
    };

    auto add_to_retset = [&](const unsigned *node_ids, const uint64_t n_ids, float *dists) {
      for (uint64_t i = 0; i < n_ids; ++i) {
#ifdef DGAI_STRICT_GRAPH_CHECKS
        check_graph_node_mapping(node_ids[i], id2loc(node_ids[i]), "decouple_search/seed");
#endif
        retset[cur_list_size++] = Neighbor(node_ids[i], dists[i], true);
        visited.insert(node_ids[i]);
      }
    };

    if (stats != nullptr) {
      stats->io_us = 0;
      stats->io_us1 = 0;
      stats->cpu_us = 0;
      stats->cpu_us1 = 0;
      stats->cpu_us2 = 0;
    }

    int64_t cur_beam_width = std::min(4ul, beam_width);
    std::vector<uint32_t> live_ids(mem_L);
    if (mem_L) {
      const uint32_t n_mem =
          mem_search_ids_with_pq(query, query_buf, mem_L, live_ids.data(), dist_scratch);
      const uint32_t n_live = n_mem;
      if (n_live > 0) {
        add_to_retset(live_ids.data(), std::min((uint64_t) n_live, l_search), dist_scratch);
      } else {
        nbr_handler->compute_dists(query_buf, &meta_.entry_point_id, 1);
        add_to_retset(&meta_.entry_point_id, 1, dist_scratch);
      }
    } else {
#ifdef OVERLAP_INIT
      nbr_handler->initialize_query(query, query_buf);
#endif
      nbr_handler->compute_dists(query_buf, &meta_.entry_point_id, 1);
      add_to_retset(&meta_.entry_point_id, 1, dist_scratch);
    }
    std::sort(retset.begin(), retset.begin() + cur_list_size);

    // Map from page_id to its I/O state: when several nodes live on the same
    // page only one read is issued and the rest reuse the buffer. To keep that
    // reuse safe under concurrent split/relocation, each entry also records the
    // loc2id of the page's slots at read time (see PageIOInfo::slot_owners).
    std::unordered_map<unsigned, PageIOInfo> page_io_map;

    // Nodes whose graph I/O has completed, waiting to be expanded.
    std::queue<ReadyEntry> ready_queue;
    // Nodes already expanded (their neighbors pushed into retset).  Prevents the
    // speculative path and the I/O-completion path from expanding the same node twice.
    tsl::robin_set<unsigned> expanded_set;
    // Page IDs whose buffer is already resident in this query's sector scratch.
    // Enables speculative in-memory expansion of co-resident nodes without extra I/O.
    tsl::robin_set<unsigned> cached_pages;

    // Snapshot the current owner (loc2id) of every slot on page `pid`. Read
    // directly from loc2id_ under its shared lock (tolerating not-yet-allocated
    // trailing slots) so we never crash on an out-of-range loc.
    auto capture_slot_owners = [&](unsigned pid) -> std::vector<uint32_t> {
      const uint32_t nslots = (uint32_t) nnodes_per_sector_graph_;
      std::vector<uint32_t> owners(nslots == 0 ? 1 : nslots, kInvalidID);
      if (nslots == 0) {
        return owners;  // one node per page: no cross-node reuse, nothing to validate.
      }
      const uint64_t base = (uint64_t) pid * nnodes_per_sector_graph_;
      loc2id_resize_mu_.lock_shared();
      const uint64_t sz = loc2id_.size();
      for (uint32_t s = 0; s < nslots; ++s) {
        const uint64_t l = base + s;
        owners[s] = (l < sz) ? loc2id_[l] : kInvalidID;
      }
      loc2id_resize_mu_.unlock_shared();
      return owners;
    };

    // Is page `pid`'s cached buffer valid for serving `id` at `loc`? Only when
    // `id` owned `loc` at the time the page was read (otherwise the snapshot
    // predates this node being relocated here).
    auto snapshot_valid_for = [&](const PageIOInfo &info, uint32_t id, unsigned loc) -> bool {
      if (nnodes_per_sector_graph_ == 0) {
        return true;  // single node per page; no aliasing possible.
      }
      const uint32_t slot = (uint32_t) loc - info.base_loc;
      return slot < info.slot_owners.size() && info.slot_owners[slot] == id;
    };

#ifdef DGAI_STRICT_GRAPH_CHECKS
    // Debug counters for stale same-page IO dedup (post-split relocation).
    uint64_t stale_snapshot_rereads = 0;
    auto warn_stale_page_snapshot = [&](uint32_t id, unsigned loc, unsigned pid, const PageIOInfo &info) {
      stale_snapshot_rereads++;
      const uint32_t slot = (uint32_t) loc - info.base_loc;
      const uint32_t owner_at_read =
          slot < info.slot_owners.size() ? info.slot_owners[slot] : kInvalidID;
      LOG(WARNING) << "decouple_search: page IO reuse skipped, node not in cached snapshot"
                   << " (likely relocated by split); re-reading"
                   << " id=" << id << " loc=" << loc << " page=" << pid << " slot=" << slot
                   << " owner_at_read=" << owner_at_read
                   << " query_stale_reread#" << stale_snapshot_rereads;
    };
#endif

    auto poll_ios = [&]() {
      if (insert_ctx != nullptr) {
        graph_reader_->poll_alloc(ctx, &insert_ctx->page_ref);
      } else {
        graph_reader_->poll_all(ctx);
      }
    };

    // Map from node ID to graph node data (neighbors only, no coords). Views
    // alias the sector buffer; safe here because MAX_N_SECTOR_READS is large
    // enough that a query does not recycle buffers, and page reuse correctness
    // is enforced by slot_owners / snapshot_valid_for above.
    std::unordered_map<unsigned, DiskGraphNode<T>> id_graph_buf_map;
    // false => torn under seqlock; caller must re-issue via send_read_no_alloc.
    // Never sync-read on the async AIO ctx: that steals in-flight completions
    // and leaves while(!on_flight.empty()) poll_all() spinning forever.
    auto process_completed_io = [&](decouple_io_t &io) -> bool {
      const unsigned loc = io.loc;
#ifdef DGAI_STRICT_GRAPH_CHECKS
      check_graph_loc_stable(io.nbr.id, id2loc(io.nbr.id), io.loc, "decouple_search/poll_all");
#endif
#ifndef DGAI_MERGE_COORD_EXCLUSIVE
      auto pit = page_io_map.find(io.page_id);
      // Missing entry => a sibling completion already declared this page torn.
      if (pit == page_io_map.end()) {
        unlock_idx(idx_lock_table, io.nbr.id);
        return false;
      }
      std::atomic_thread_fence(std::memory_order_acquire);
      const uint32_t v1 = pit->second.seq_v1;
      const uint32_t v2 = page_ver_load(io.page_id);
      if ((v1 & 1) || v1 != v2) {
        page_io_map.erase(pit);
        unlock_idx(idx_lock_table, io.nbr.id);
        return false;
      }
#endif
      id_graph_buf_map.emplace(
          std::piecewise_construct, std::forward_as_tuple(io.nbr.id),
          std::forward_as_tuple((char *) io.read_req->buf, loc, max_graph_node_len_, nnodes_per_sector_graph_,
                                meta_.range));
#ifdef DGAI_STRICT_GRAPH_CHECKS
      check_graph_nnbrs(id_graph_buf_map.at(io.nbr.id).nnbrs, io.nbr.id, loc, "decouple_search/poll_all");
#endif
      unlock_idx(idx_lock_table, io.nbr.id);
      return true;
    };

#ifdef USE_PRIORITY_QUEUE
    // Priority-queue container (vector + heap) for in-flight I/O entries.
    // Using explicit heap operations so we can rebuild after poll_all()
    // when completion status changes externally via read_req->finished.
    std::vector<decouple_io_t> on_flight_ios;
    // Global timestamp incremented on every send_read_req call to guarantee
    // FIFO ordering among same-page requests.
    uint64_t io_timestamp = 0;

    auto send_read_req = [&](Neighbor &item, unsigned rk) -> bool {
      item.flag = false;

      // Lock before id2loc (same as pipe_search/beam_search) so split/insert
      // cannot change the mapping between lookup and I/O.
      lock_idx(idx_lock_table, kInvalidID, std::vector<uint32_t>{item.id}, true);
      const unsigned loc = id2loc(item.id);
#ifdef DGAI_STRICT_GRAPH_CHECKS
      check_graph_node_mapping(item.id, loc, "decouple_search/send_read_req");
#endif
      const unsigned pid = graph_loc_sector_no(loc);

      // Reuse the page's I/O if it is already issued AND its snapshot still
      // matches this node's loc. A concurrent split may have relocated this node
      // onto the page after it was read; in that (rare) case the cached buffer
      // is stale for us, so we fall through and re-read once.
      auto it = page_io_map.find(pid);
      if (it != page_io_map.end()) {
        if (snapshot_valid_for(it->second, item.id, loc)) {
          decouple_io_t io{item, pid, loc, it->second.req, it->second.rk, io_timestamp++};
          on_flight_ios.push_back(io);
          std::push_heap(on_flight_ios.begin(), on_flight_ios.end(), DecoupleIOCompare());
          return true;
        }
#ifdef DGAI_STRICT_GRAPH_CHECKS
        warn_stale_page_snapshot(item.id, loc, pid, it->second);
#endif
      }

      // Issue I/O: first read for this page, or a re-read after a stale-snapshot
      // miss (which replaces the page_io_map entry and its owner snapshot).
      uint64_t &cur_buf_idx = query_buf->sector_idx;
      auto buf = sector_scratch + cur_buf_idx * size_per_io;
      auto &req = query_buf->reqs[cur_buf_idx];
      req = IORequest(static_cast<uint64_t>(pid) * SECTOR_LEN, size_per_io_graph_, buf,
                      graph_u_loc_offset(loc), max_graph_node_len_, sector_scratch);
      // Sample seqlock + slot owners before issuing IO so the snapshot matches
      // the buffer contents under concurrent merge/compaction writes.
      const uint32_t seq_v1 = page_ver_load(pid);
      auto owners = capture_slot_owners(pid);
      graph_reader_->send_read_no_alloc(req, ctx);

      page_io_map[pid] = {&req, rk, (uint32_t) (pid * nnodes_per_sector_graph_), std::move(owners), seq_v1};

      decouple_io_t io{item, pid, loc, &req, rk, io_timestamp++};
      on_flight_ios.push_back(io);
      std::push_heap(on_flight_ios.begin(), on_flight_ios.end(), DecoupleIOCompare());

      cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;

      if (stats != nullptr) {
        stats->n_ios++;
      }
      return true;
    };

    auto poll_all = [&]() -> std::pair<int, int> {
      poll_ios();
      if (!on_flight_ios.empty()) {
        std::make_heap(on_flight_ios.begin(), on_flight_ios.end(), DecoupleIOCompare());
      }
      unsigned n_in = 0, n_out = 0;
      while (!on_flight_ios.empty() && on_flight_ios.front().finished()) {
        decouple_io_t io = on_flight_ios.front();
        std::pop_heap(on_flight_ios.begin(), on_flight_ios.end(), DecoupleIOCompare());
        on_flight_ios.pop_back();

        if (!process_completed_io(io)) {
          // Torn page: re-issue async read (do not sync-read on this ctx).
          Neighbor retry_nbr = io.nbr;
          send_read_req(retry_nbr, io.rk);
          continue;
        }
        // Queue for expansion and mark the page resident for speculative reuse.
        ready_queue.push({io.nbr.id, io.nbr.distance});
        cached_pages.insert(io.page_id);
        unsigned bound = std::min<unsigned>(cur_list_size, (unsigned) l_search);
        io.nbr.distance <= retset[bound - 1].distance ? ++n_in : ++n_out;
      }
      return std::make_pair(n_in, n_out);
    };

    auto send_best_read_req = [&](uint32_t n) -> bool {
      unsigned n_sent = 0, marker = 0;
      const unsigned frontier = std::min<unsigned>(cur_list_size, (unsigned) l_search);
      while (marker < frontier && n_sent < n) {
        while (marker < frontier &&
               (retset[marker].flag == false ||
                id_graph_buf_map.find(retset[marker].id) != id_graph_buf_map.end())) {
          retset[marker].flag = false;
          ++marker;
        }
        if (marker >= frontier) {
          break;
        }
        n_sent += send_read_req(retset[marker], marker);
      }
      return n_sent != 0;
    };
#else
    std::queue<decouple_io_t> on_flight_ios;
    auto send_read_req = [&](Neighbor &item) -> bool {
      item.flag = false;

      lock_idx(idx_lock_table, kInvalidID, std::vector<uint32_t>{item.id}, true);
      const unsigned loc = id2loc(item.id);
#ifdef DGAI_STRICT_GRAPH_CHECKS
      check_graph_node_mapping(item.id, loc, "decouple_search/send_read_req");
#endif
      const unsigned pid = graph_loc_sector_no(loc);

      auto it = page_io_map.find(pid);
      if (it != page_io_map.end()) {
        if (snapshot_valid_for(it->second, item.id, loc)) {
          on_flight_ios.push(decouple_io_t{item, pid, loc, it->second.req});
          return true;
        }
#ifdef DGAI_STRICT_GRAPH_CHECKS
        warn_stale_page_snapshot(item.id, loc, pid, it->second);
#endif
      }

      uint64_t &cur_buf_idx = query_buf->sector_idx;
      auto buf = sector_scratch + cur_buf_idx * size_per_io;
      auto &req = query_buf->reqs[cur_buf_idx];
      req = IORequest(static_cast<uint64_t>(pid) * SECTOR_LEN, size_per_io_graph_, buf,
                      graph_u_loc_offset(loc), max_graph_node_len_, sector_scratch);
      const uint32_t seq_v1 = page_ver_load(pid);
      auto owners = capture_slot_owners(pid);
      graph_reader_->send_read_no_alloc(req, ctx);

      page_io_map[pid] = {&req, 0, (uint32_t) (pid * nnodes_per_sector_graph_), std::move(owners), seq_v1};

      on_flight_ios.push(decouple_io_t{item, pid, loc, &req});
      cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;

      if (stats != nullptr) {
        stats->n_ios++;
      }
      return true;
    };

    auto poll_all = [&]() -> std::pair<int, int> {
      poll_ios();
      unsigned n_in = 0, n_out = 0;
      while (!on_flight_ios.empty() && on_flight_ios.front().finished()) {
        decouple_io_t io = on_flight_ios.front();
        on_flight_ios.pop();
        if (!process_completed_io(io)) {
          Neighbor retry_nbr = io.nbr;
          send_read_req(retry_nbr);
          continue;
        }
        // Queue for expansion and mark the page resident for speculative reuse.
        ready_queue.push({io.nbr.id, io.nbr.distance});
        cached_pages.insert(io.page_id);
        unsigned bound = std::min<unsigned>(cur_list_size, (unsigned) l_search);
        io.nbr.distance <= retset[bound - 1].distance ? ++n_in : ++n_out;
      }
      return std::make_pair(n_in, n_out);
    };

    auto send_best_read_req = [&](uint32_t n) -> bool {
      unsigned n_sent = 0, marker = 0;
      const unsigned frontier = std::min<unsigned>(cur_list_size, (unsigned) l_search);
      while (marker < frontier && n_sent < n) {
        while (marker < frontier &&
               (retset[marker].flag == false ||
                id_graph_buf_map.find(retset[marker].id) != id_graph_buf_map.end())) {
          retset[marker].flag = false;
          ++marker;
        }
        if (marker >= frontier) {
          break;
        }
        n_sent += send_read_req(retset[marker]);
      }
      return n_sent != 0;
    };
#endif

    // Expand one node whose graph data is resident, marking it visited/expanded
    // and pushing its neighbors into retset.  `node` points into a resident page
    // buffer, `loc` is its location.  Returns false if the node was already
    // expanded.  Used by the speculative in-memory path (calc_memory).
    auto activate_node = [&](DiskGraphNode<T> &node, unsigned loc, int cache_pos = -1) -> bool {
      unsigned id = loc2id(loc);
      if (expanded_set.find(id) != expanded_set.end())
        return false;

      float dist = 0.0f;
      if (cache_pos == -1) {
        for (unsigned i = 0; i < cur_list_size; ++i) {
          if (retset[i].id == id) {
            dist = retset[i].distance;
            retset[i].flag = false;
            retset[i].visited = true;
            break;
          }
        }
      } else {
        dist = retset[cache_pos].distance;
        retset[cache_pos].flag = false;
        retset[cache_pos].visited = true;
      }

      expanded_set.insert(id);
      record_visited_node(id, dist);
      unsigned nk = cur_list_size;
      compute_and_push_nbrs(node, id, nk);
      return true;
    };

    // Speculative in-memory expansion: while real I/O is in flight, look for an
    // unvisited retset node whose page is already resident and expand it for
    // free (no extra I/O).  A brief idx read-lock guards against a concurrent
    // page split relocating the node mid-read.
    auto calc_cache = [&]() -> bool {
      for (unsigned i = 0; i < cur_list_size; ++i) {
        if (retset[i].visited)
          continue;
        const uint32_t id = retset[i].id;
        unsigned loc = id2loc(id);
        unsigned pid = graph_loc_sector_no(loc);
        if (cached_pages.find(pid) == cached_pages.end())
          continue;  // page not resident: cheap check, no lock taken

        // Page completed => all of its idx locks were already released in
        // poll_all, so locking this id here cannot self-deadlock.
        auto locked = lock_idx(idx_lock_table, kInvalidID, std::vector<uint32_t>{id}, true);
        // Re-validate under the lock (a split may have moved the node).
        loc = id2loc(id);
        pid = graph_loc_sector_no(loc);
        if (cached_pages.find(pid) == cached_pages.end()) {
          unlock_idx(idx_lock_table, locked);
          continue;
        }
        auto info_it = page_io_map.find(pid);
        if (info_it == page_io_map.end()) {
          unlock_idx(idx_lock_table, locked);
          continue;
        }
        // A re-read may replace this page's buffer while cached_pages still
        // records an older completion. Only expand a finished, matching view.
        if (!info_it->second.req->finished || !snapshot_valid_for(info_it->second, id, loc)) {
          unlock_idx(idx_lock_table, locked);
          continue;
        }

        auto map_it = id_graph_buf_map.find(id);
        if (map_it == id_graph_buf_map.end()) {
          DiskGraphNode<T> node_from_page = graph_node_from_page((char *) info_it->second.req->buf, loc);
          map_it = id_graph_buf_map.insert(std::make_pair(id, node_from_page)).first;
        }
        DiskGraphNode<T> &node = map_it->second;
        bool ok = activate_node(node, loc, i);
        unlock_idx(idx_lock_table, locked);
        if (ok)
          return true;
      }
      return false;
    };

    // Expand a node whose I/O just completed (its graph data is in id_graph_buf_map).
    auto expand_one = [&](unsigned id, float distance) -> bool {
      if (expanded_set.find(id) != expanded_set.end())
        return false;  // already expanded speculatively
      for (unsigned i = 0; i < cur_list_size; ++i) {
        if (retset[i].id == id && !retset[i].visited) {
          retset[i].flag = false;
          retset[i].visited = true;
          expanded_set.insert(id);
          auto it = id_graph_buf_map.find(id);
          if (it == id_graph_buf_map.end())
            return false;
          auto &[_, node] = *it;
          record_visited_node(id, distance);
          unsigned nk = cur_list_size;
          compute_and_push_nbrs(node, id, nk);
          return true;
        }
      }
      return false;
    };

    // Index of the first unvisited node that has an I/O in flight but whose
    // graph data is not yet ready (used to drive the adaptive beam width).
    auto find_first_unvisited_eager = [&]() -> unsigned {
      const unsigned frontier = std::min<unsigned>(cur_list_size, (unsigned) l_search);
      unsigned first_unvisited_eager = frontier;
      for (unsigned i = 0; i < frontier; ++i) {
        if (!retset[i].visited && retset[i].flag &&
            id_graph_buf_map.find(retset[i].id) == id_graph_buf_map.end()) {
          first_unvisited_eager = i;
          break;
        }
      }
      return first_unvisited_eager;
    };

    auto get_first_unvisited = [&]() -> int {
      int ret = -1;
      const unsigned frontier = std::min<unsigned>(cur_list_size, (unsigned) l_search);
      for (unsigned i = 0; i < frontier; ++i) {
        if (!retset[i].visited) {
          ret = i;
          break;
        }
      }
      return ret;
    };

    auto cpu2_st = std::chrono::high_resolution_clock::now();
    send_best_read_req(cur_beam_width - on_flight_ios.size());
    unsigned max_marker = 0;

    int cur_n_in = 0, cur_tot = 0;
    while (get_first_unvisited() != -1) {
      Timer io_timer;
      auto [n_in, n_out] = poll_all();
      if (stats != nullptr) {
        stats->io_us += (double) io_timer.elapsed();
      }
      std::ignore = n_in;
      std::ignore = n_out;

      if (max_marker >= 5 && n_in + n_out > 0) {
        cur_n_in += n_in;
        cur_tot += n_in + n_out;
        constexpr double kWasteThreshold = 0.1;
        if ((cur_tot - cur_n_in) * 1.0 / cur_tot <= kWasteThreshold) {
          cur_beam_width = cur_beam_width + 1;
          cur_beam_width = std::max(cur_beam_width, 4l);
          cur_beam_width = std::min((int64_t) beam_width, cur_beam_width);
        }

        // (relaxed_monotonicity_l has been removed)
      }


      // --- Expand one node: prefer a node whose I/O completed; otherwise
      //     speculatively expand a resident-page node to overlap CPU with I/O ---
      if (!ready_queue.empty()) {
        auto e = ready_queue.front();
        ready_queue.pop();
        expand_one(e.id, e.distance);
      } else if (max_marker < 10) {
#ifdef CACHE_N
        calc_cache();
#endif
      } else if ((int64_t) on_flight_ios.size() == cur_beam_width) {
#ifdef CACHE_N
        calc_cache();
#endif
      }

      io_timer.reset();
      if ((int64_t) on_flight_ios.size() < cur_beam_width) {
        send_best_read_req(1);
      }
      if (stats != nullptr) {
        stats->io_us += (double) io_timer.elapsed();
      }


      // --- Track the deepest eager frontier reached (drives adaptive width) ---
      if (max_marker < 5) {
        max_marker = std::max(max_marker, find_first_unvisited_eager());
      }
    }

    // Drain remaining on-flight IOs and unlock their idx locks.
    // Must always drain to release idx locks and populate page cache.
    {
      int drain_spins = 0;
      while (!on_flight_ios.empty()) {
        Timer io_timer;
        std::ignore = poll_all();
        if (stats != nullptr) {
          stats->io_us += (double) io_timer.elapsed();
        }
        // Safety: never spin forever if an AIO completion was lost.
        if (++drain_spins > 1000000) {
          LOG(ERROR) << "decouple_search: on_flight drain timeout, forcing unlock n="
                     << on_flight_ios.size();
#ifdef USE_PRIORITY_QUEUE
          for (auto &io : on_flight_ios) unlock_idx(idx_lock_table, io.nbr.id);
          on_flight_ios.clear();
#else
          while (!on_flight_ios.empty()) {
            unlock_idx(idx_lock_table, on_flight_ios.front().nbr.id);
            on_flight_ios.pop();
          }
#endif
          break;
        }
      }
      for (unsigned i = 0; i < cur_list_size; ++i) {
        if (!retset[i].visited && id_graph_buf_map.find(retset[i].id) != id_graph_buf_map.end()) {
          retset[i].visited = true;
          record_visited_node(retset[i].id, retset[i].distance);
        }
      }
    }
    // Keep the frozen entry available during traversal and insertion, but do
    // not let a routing-only node consume a query result or reranking slot.
    if (insert_ctx == nullptr) {
      full_retset.erase(std::remove_if(full_retset.begin(), full_retset.end(),
                                     [this](const Neighbor &n) { return is_routing_only(n.id); }),
                        full_retset.end());
    }
    // --- Phase 2: Read exact vectors and compute exact distances for reranking ---
    auto rerank_st = std::chrono::high_resolution_clock::now();
    // Vectors are packed records (same pattern as original), no sector 0.
    // id == loc for ordering; use vec_loc_sector_no / vec_sector_offset for physical access.
    // Sort by approximate (PQ) distance so the most promising candidates are first.
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

#ifdef HEURISTIC_RERANK
    //cmake -DCMAKE_CXX_FLAGS="-DHEURISTIC_RERANK"
    const float kEps = 0.1f;
    const int kBeta = 2;
    const size_t kPipelineBatch = 10;
    const size_t kPrefetchDepth = 3;
    const float kSoftStopThreshold = 0.20f;
    const float kMinPrefetchDelta = 0.30f;

    struct PrefetchBatch {
      size_t io_start;
      size_t io_count;
      std::vector<IORequest> reqs;
    };

    std::vector<Neighbor> &unique_candidates = full_retset;
    uint64_t vec_data_len = meta_.data_dim * sizeof(T);
    void *vec_ctx = vector_reader_->get_ctx();

    const uint64_t batch_sz = kPipelineBatch > 0 ? (uint64_t)kPipelineBatch : 1;
    int stable_cnt = 0;
    uint64_t n_read = 0;

    std::vector<Neighbor> Q(k_search + 1);
    unsigned q_sz = 0;
    std::vector<unsigned> S_prev, S_curr;
    S_prev.reserve(k_search);
    S_curr.reserve(k_search);

    std::deque<PrefetchBatch> io_inflight;
    size_t io_next_submit = 0;
    bool soft_stop = false;
    bool hard_stop = false;

    auto slot_for = [&](size_t j) { return j % MAX_N_SECTOR_READS; };

    auto submit_async_io = [&](size_t start, size_t count) {
      PrefetchBatch batch;
      batch.io_start = start;
      batch.io_count = count;
      batch.reqs.reserve(count);
      for (size_t j = start; j < start + count; ++j) {
        uint32_t id = unique_candidates[j].id;
        uint64_t sector_no = vec_loc_sector_no(id);
        uint64_t u_off = vec_sector_offset(id);
        size_t slot = slot_for(j);
        char *buf = sector_scratch + slot * size_per_io_vec_;
        batch.reqs.push_back(IORequest(sector_no * SECTOR_LEN, size_per_io_vec_, buf,
                                       u_off, vec_data_len, sector_scratch));
        query_buf->reqs[slot] = batch.reqs.back();
      }
      vector_reader_->send_read_no_alloc(batch.reqs, vec_ctx);
      io_inflight.push_back(std::move(batch));
    };

    auto wait_for_range = [&](size_t start, size_t end) {
      while (true) {
        vector_reader_->poll_all(vec_ctx);
        for (auto &b : io_inflight)
          for (size_t k = 0; k < b.io_count; ++k)
            query_buf->reqs[slot_for(b.io_start + k)].finished = b.reqs[k].finished;
        bool all_done = true;
        for (size_t j = start; j < end; ++j) {
          if (!query_buf->reqs[slot_for(j)].finished) {
            all_done = false;
            break;
          }
        }
        if (all_done) break;
        vector_reader_->poll_wait(vec_ctx);
      }
    };

    // Prime the I/O pipeline.
    size_t total_candidates = unique_candidates.size();
    for (size_t d = 0; d < kPrefetchDepth && io_next_submit < total_candidates; ++d) {
      size_t count = std::min(kPipelineBatch, total_candidates - io_next_submit);
      submit_async_io(io_next_submit, count);
      io_next_submit += count;
    }

    // Stability loop.
    for (size_t i = 0; i < total_candidates && !hard_stop; i += batch_sz) {
      size_t batch_end = std::min(i + batch_sz, total_candidates);
      wait_for_range(i, batch_end);

      S_prev.clear();
      for (unsigned p = 0; p < q_sz; ++p) S_prev.push_back(Q[p].id);

      for (size_t j = i; j < batch_end; ++j) {
        uint32_t id = unique_candidates[j].id;
        auto &req = query_buf->reqs[slot_for(j)];
        const T *raw_vec = (const T *)((char *)req.buf + req.u_offset);
        float d = dist_cmp->compare(query, raw_vec, (unsigned)aligned_dim);
        unique_candidates[j].distance = d;

        if (q_sz == 0) {
          Q[0] = Neighbor(id, d, true); q_sz = 1;
        } else if (q_sz < k_search) {
          InsertIntoPool(Q.data(), q_sz, Neighbor(id, d, true)); q_sz++;
        } else {
          InsertIntoPool(Q.data(), q_sz, Neighbor(id, d, true));
        }

        if (insert_ctx != nullptr) {
          if (unlikely(n_read >= static_cast<uint64_t>(this->params.L) * INSERT_COORD_BUF_L_FACTOR)) {
            LOG(ERROR) << "Please allocate larger coord_buf.";
            crash();
          }
          T *coord_ptr = insert_ctx->coord_buf + n_read * aligned_dim;
          memcpy(coord_ptr, raw_vec, meta_.data_dim * sizeof(T));
          insert_ctx->coord_map.insert(std::make_pair(id, coord_ptr));
        }
        n_read++;
      }

      S_curr.clear();
      for (unsigned p = 0; p < q_sz; ++p) S_curr.push_back(Q[p].id);
      unsigned changed = 0;
      for (unsigned c = 0; c < S_curr.size(); ++c) {
        bool found = false;
        for (unsigned p = 0; p < S_prev.size(); ++p) {
          if (S_prev[p] == S_curr[c]) { found = true; break; }
        }
        if (!found) changed++;
      }
      float delta = (k_search > 0) ? (float)changed / (float)k_search : 0.0f;

      if (q_sz >= k_search) {
        if (delta <= kEps) {
          stable_cnt++;
          if (stable_cnt >= kBeta) { hard_stop = true; }
        } else {
          stable_cnt = 0;
        }
        if (!soft_stop && delta <= kSoftStopThreshold) {
          soft_stop = true;
        } else if (soft_stop && delta > kMinPrefetchDelta) {
          soft_stop = false;
        }
      }

      while (!io_inflight.empty() &&
             io_inflight.front().io_start + io_inflight.front().io_count <= batch_end) {
        io_inflight.pop_front();
      }
      size_t max_inflight = soft_stop ? 1 : kPrefetchDepth;
      while (!hard_stop && io_inflight.size() < max_inflight && io_next_submit < total_candidates) {
        size_t count = std::min(kPipelineBatch, total_candidates - io_next_submit);
        submit_async_io(io_next_submit, count);
        io_next_submit += count;
        if (soft_stop) break;
      }
    }

    // Drain remaining inflight I/Os.
    while (!io_inflight.empty()) {
      vector_reader_->poll_all(vec_ctx);
      for (auto &b : io_inflight)
        for (size_t k = 0; k < b.io_count; ++k)
          query_buf->reqs[slot_for(b.io_start + k)].finished = b.reqs[k].finished;
      while (!io_inflight.empty()) {
        bool all_done = true;
        for (size_t k = 0; k < io_inflight.front().io_count; ++k)
          if (!query_buf->reqs[slot_for(io_inflight.front().io_start + k)].finished)
            { all_done = false; break; }
        if (all_done) io_inflight.pop_front(); else break;
      }
      if (!io_inflight.empty()) vector_reader_->poll_wait(vec_ctx);
    }

    std::sort(unique_candidates.begin(), unique_candidates.begin() + n_read,
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    expanded_nodes_info.assign(unique_candidates.begin(),
                               unique_candidates.begin() + n_read);

    auto cpu2_ed = std::chrono::high_resolution_clock::now();
    if (stats != nullptr) {
      stats->cpu_us2 = std::chrono::duration_cast<std::chrono::microseconds>(cpu2_ed - cpu2_st).count();
      stats->cpu_us = n_computes;
      stats->rerank_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu2_ed - rerank_st).count();
    }

    // Copy top k_search results.
    uint64_t t = 0;
    if (res_tags != nullptr) {
      for (uint64_t i = 0; i < expanded_nodes_info.size() && t < k_search; i++) {
        if (is_routing_only(expanded_nodes_info[i].id)) continue;
        res_tags[t] = id2tag(expanded_nodes_info[i].id);
        if (distances != nullptr) {
          distances[t] = expanded_nodes_info[i].distance;
        }
        t++;
      }
    }

    push_query_buf(query_buf);
    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
    return t;
#else
    if (full_retset.size() > active_l_search) {
      full_retset.resize(active_l_search);
    }

    // Exact-vector rerank: window of size W = rerank_window around K.
    //   [max(0, K - W/2), min(L, K + W/2))
    // Insert needs exact coords for every returned candidate, so always start at 0.
    const size_t L = full_retset.size();
    const size_t W = std::max<size_t>(1, static_cast<size_t>(rerank_window));
    const size_t half = W / 2;
    size_t rerank_start =
        (insert_ctx != nullptr || k_search <= half) ? 0 : static_cast<size_t>(k_search - half);
    size_t rerank_end = std::min(L, static_cast<size_t>(k_search + half));
    if (insert_ctx != nullptr) {
      // Cover the full insert candidate list (already capped at l_search above).
      rerank_end = L;
    }
    if (rerank_start > rerank_end) {
      rerank_start = rerank_end;
    }
    full_retset.resize(rerank_end);

    // Each node is recorded only once (guarded by visited), so dedup is unnecessary;
    // full_retset is used directly as the candidate list for exact-vector reads.
    std::vector<Neighbor> &unique_candidates = full_retset;

    // Read exact vectors using AIO via vector_reader_.
    uint64_t vec_data_len = meta_.data_dim * sizeof(T);
    void *vec_ctx = vector_reader_->get_ctx();

    // // Lock all candidate nodes for vector reads.
    // for (auto &c : unique_candidates) {
    //   lock_vec(vec_lock_table, c.id, std::vector<uint32_t>(), true);
    // }

    // Process only the rerank window in batches of MAX_IO_BATCH.
    for (size_t batch_start = rerank_start; batch_start < unique_candidates.size();
         batch_start += MAX_IO_BATCH) {
      size_t batch_end = std::min(batch_start + MAX_IO_BATCH, unique_candidates.size());
      size_t batch_size = batch_end - batch_start;

      // Resolve overlay (still-staged) vectors first, then disk-read the rest. Overlay
      // misses are guaranteed on disk (combiner erases only after the disk write), so a
      // miss here means the disk read below sees the correct, immutable vector bytes.
      std::vector<IORequest> batch_reads;
      batch_reads.reserve(batch_size);
      for (size_t i = batch_start; i < batch_end; ++i) {
        uint32_t id = unique_candidates[i].id;
        uint64_t sector_no = vec_loc_sector_no(id);
        uint64_t u_off = vec_sector_offset(id);

        size_t slot = i - batch_start;
        char *buf = sector_scratch + slot * size_per_io_vec_;
        if (try_read_decouple_vector(id, buf + u_off)) {
          continue;
        }
        batch_reads.push_back(IORequest(sector_no * SECTOR_LEN, size_per_io_vec_, buf,
                                        u_off, vec_data_len, sector_scratch));
      }
      if (!batch_reads.empty()) {
        Timer io_timer;
        vector_reader_->read(batch_reads, vec_ctx);
        if (stats != nullptr) {
          stats->io_us += (double) io_timer.elapsed();
        }
      }

      for (size_t i = 0; i < batch_size; ++i) {
        uint32_t id = unique_candidates[batch_start + i].id;
        uint64_t u_off = vec_sector_offset(id);
        char *buf = sector_scratch + i * size_per_io_vec_;
        const T *raw_vec = (const T *) (buf + u_off);
        float exact_dist = dist_cmp->compare(query, raw_vec, (unsigned) aligned_dim);
        unique_candidates[batch_start + i].distance = exact_dist;

        if (insert_ctx != nullptr) {
          const uint64_t coord_cap = static_cast<uint64_t>(this->params.L) * INSERT_COORD_BUF_L_FACTOR;
          uint64_t idx = batch_start + i;
          if (unlikely(idx >= coord_cap)) {
            LOG(ERROR) << "Please allocate larger coord_buf. idx=" << idx << " cap=" << coord_cap;
            crash();
          }
          T *coord_ptr = insert_ctx->coord_buf + idx * aligned_dim;
          memcpy(coord_ptr, raw_vec, meta_.data_dim * sizeof(T));
          insert_ctx->coord_map.insert(std::make_pair(id, coord_ptr));
        }
      }
    }

    // // Unlock vector reads.
    // for (auto &c : unique_candidates) {
    //   unlock_vec(vec_lock_table, c.id, std::vector<uint32_t>());
    // }

    // Sort only the reranked window by exact distance; prefix stays in approx order.
    std::sort(unique_candidates.begin() + static_cast<std::ptrdiff_t>(rerank_start),
              unique_candidates.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    // Return the full expanded node list to the caller.
    expanded_nodes_info.assign(unique_candidates.begin(), unique_candidates.end());

    auto cpu2_ed = std::chrono::high_resolution_clock::now();
    if (stats != nullptr) {
      stats->cpu_us2 = std::chrono::duration_cast<std::chrono::microseconds>(cpu2_ed - cpu2_st).count();
      stats->cpu_us = n_computes;
      stats->rerank_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu2_ed - rerank_st).count();
    }

    // Copy top k_search results (skip if caller doesn't need them, e.g. insert path).
    uint64_t t = 0;
    if (res_tags != nullptr) {
      for (uint64_t i = 0; i < unique_candidates.size() && t < k_search; i++) {
        if (is_routing_only(unique_candidates[i].id)) continue;
        res_tags[t] = id2tag(unique_candidates[i].id);
        if (distances != nullptr) {
          distances[t] = unique_candidates[i].distance;
        }
        t++;
      }
    }

    push_query_buf(query_buf);
#endif

#ifdef DGAI_STRICT_GRAPH_CHECKS
    if (stale_snapshot_rereads > 0) {
      LOG(WARNING) << "decouple_search: query finished with " << stale_snapshot_rereads
                   << " stale page snapshot re-read(s) (IO dedup invalidated by split/relocation)";
    }
#endif

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
    return t;
  }

  // =========================================================================
  // Non-decoupled single-stage search.
  // Reads full pages (graph + coords) via reader. Exact distances from coords
  // computed during traversal — no Phase 2 reranking needed.
  // =========================================================================
  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::non_decouple_search(const T *query1, const uint64_t k_search, const uint32_t mem_L,
                                                const uint64_t l_search, std::vector<Neighbor> &expanded_nodes_info,
                                                const uint64_t beam_width, QueryStats *stats,
                                                tsl::robin_map<uint32_t, T *> *coord_map,
                                                T *coord_buf,
                                                TagT *res_tags, float *distances,
                                                std::vector<uint64_t> *passthrough_page_ref,
                                                uint64_t max_candidates) {
    EpochGuard search_lifetime(reader_epoch_);
#ifndef READ_ONLY_TESTS
    std::vector<uint64_t> *page_ref = passthrough_page_ref ? passthrough_page_ref : new std::vector<uint64_t> {};
#endif
    QueryBuffer<T> *query_buf = pop_query_buf(query1);
#ifdef USE_AIO
    void *ctx = reader->get_ctx();
#else
    void *ctx = reader->get_ctx();
#endif

    if (beam_width > MAX_N_SECTOR_READS) {
      LOG(ERROR) << "Beamwidth can not be higher than MAX_N_SECTOR_READS";
      crash();
    }

    const T *query = query_buf->aligned_query_T;
    query_buf->reset();

    T *data_buf = query_buf->coord_scratch;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    char *sector_scratch = query_buf->sector_scratch;
    float *dist_scratch = query_buf->aligned_dist_scratch;

    Timer query_timer;
#ifdef CACHE_N
    const uint64_t extendL = l_search * 2;
#else
    const uint64_t extendL = l_search;
#endif
    std::vector<Neighbor> retset(mem_L + this->params.R + extendL * 10);
    auto &visited = *(query_buf->visited);
    unsigned cur_list_size = 0;

    std::vector<Neighbor> full_retset;
    full_retset.reserve(l_search * 10);

#ifndef OVERLAP_INIT
    nbr_handler->initialize_query(query, query_buf);
#endif

    // Compute exact distance from coords in the page (non-decoupled: coords + graph together)
    auto compute_exact_and_record = [&](const DiskNode<T> &node, const unsigned id) {
      T *node_fp_coords_copy = data_buf;
      memcpy(node_fp_coords_copy, node.coords, meta_.data_dim * sizeof(T));
      float exact_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
      full_retset.push_back(Neighbor(id, exact_dist, true));
    };

    uint64_t n_computes = 0;
    std::vector<unsigned> nbr_filtered;
    auto compute_and_push_nbrs = [&](DiskNode<T> &node, unsigned &nk) {
      unsigned nbors_cand_size = 0;
      nbr_filtered.resize(node.nnbrs);
      for (unsigned m = 0; m < node.nnbrs; ++m) {
        if (visited.find(node.nbrs[m]) == visited.end()) {
          nbr_filtered[nbors_cand_size++] = node.nbrs[m];
          visited.insert(node.nbrs[m]);
        }
      }

      n_computes += nbors_cand_size;
      if (nbors_cand_size) {
        nbr_handler->compute_dists(query_buf, nbr_filtered.data(), nbors_cand_size);
        for (unsigned m = 0; m < nbors_cand_size; ++m) {
          const int nbor_id = nbr_filtered[m];
          const float nbor_dist = dist_scratch[m];
          if (stats != nullptr) stats->n_cmps++;

          if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == extendL))
            continue;
          Neighbor nn(nbor_id, nbor_dist, true);
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
          if (cur_list_size < extendL) {
            ++cur_list_size;
            if (unlikely(cur_list_size >= retset.size())) retset.resize(2 * cur_list_size);
          }
          if (r < nk) nk = r;
        }
      }
    };

    auto add_to_retset = [&](const unsigned *node_ids, const uint64_t n_ids, float *dists) {
      for (uint64_t i = 0; i < n_ids; ++i) {
        retset[cur_list_size++] = Neighbor(node_ids[i], dists[i], true);
        visited.insert(node_ids[i]);
      }
    };

    if (stats != nullptr) {
      stats->io_us = 0; stats->io_us1 = 0; stats->cpu_us = 0; stats->cpu_us1 = 0; stats->cpu_us2 = 0;
    }

    int64_t cur_beam_width = std::min(4ul, beam_width);
    std::vector<uint32_t> live_ids(mem_L);
    if (mem_L) {
      const uint32_t n_mem = mem_search_ids_with_pq(query, query_buf, mem_L, live_ids.data(), dist_scratch);
      const uint32_t n_live = n_mem;
      if (n_live > 0) {
        add_to_retset(live_ids.data(), std::min((uint64_t) n_live, l_search), dist_scratch);
      } else {
        nbr_handler->compute_dists(query_buf, &meta_.entry_point_id, 1);
        add_to_retset(&meta_.entry_point_id, 1, dist_scratch);
      }
    } else {
#ifdef OVERLAP_INIT
      nbr_handler->initialize_query(query, query_buf);
#endif
      nbr_handler->compute_dists(query_buf, &meta_.entry_point_id, 1);
      add_to_retset(&meta_.entry_point_id, 1, dist_scratch);
    }
    std::sort(retset.begin(), retset.begin() + cur_list_size);

    std::unordered_map<unsigned, PageIOInfo> page_io_map;
    std::queue<ReadyEntry> ready_queue;
    tsl::robin_set<unsigned> expanded_set;
    tsl::robin_set<unsigned> cached_pages;

    auto reconcile_page_cache = [&](IORequest *req) {
#ifndef READ_ONLY_TESTS
      {
        uint64_t n_sectors = req->len / SECTOR_LEN;
        uint64_t base_sector = req->offset / SECTOR_LEN;
        for (uint64_t si = 0; si < n_sectors; si++)
          pipeann::cache.finish_read(base_sector + si, (uint8_t *)req->buf + si * SECTOR_LEN, nullptr);
      }
#endif
    };

#ifdef USE_PRIORITY_QUEUE
    std::vector<decouple_io_t> on_flight_ios;
    uint64_t io_timestamp = 0;

    auto send_read_req = [&](Neighbor &item, unsigned rk) -> bool {
      item.flag = false;
      const unsigned loc = id2loc(item.id);
      const unsigned pid = loc_sector_no(loc);
      lock_idx(idx_lock_table, kInvalidID, std::vector<uint32_t>{item.id}, true);

      auto it = page_io_map.find(pid);
      if (it != page_io_map.end()) {
        on_flight_ios.push_back({item, pid, loc, it->second.req, it->second.rk, io_timestamp++});
        std::push_heap(on_flight_ios.begin(), on_flight_ios.end(), DecoupleIOCompare());
#ifndef READ_ONLY_TESTS
        page_ref->push_back(pid);
#endif
        return true;
      }

      uint64_t &cur_buf_idx = query_buf->sector_idx;
      auto buf = sector_scratch + cur_buf_idx * size_per_io;
      auto &req = query_buf->reqs[cur_buf_idx];
      req = IORequest(static_cast<uint64_t>(pid) * SECTOR_LEN, size_per_io, buf,
                      u_loc_offset(loc), meta_.max_node_len, sector_scratch);
      reader->send_read_no_alloc(req, ctx);
#ifndef READ_ONLY_TESTS
      page_ref->push_back(pid);
#endif
      page_io_map[pid] = {&req, rk};
      on_flight_ios.push_back({item, pid, loc, &req, rk, io_timestamp++});
      std::push_heap(on_flight_ios.begin(), on_flight_ios.end(), DecoupleIOCompare());
      cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;
      if (stats != nullptr) stats->n_ios++;
      return true;
    };

    std::unordered_map<unsigned, DiskNode<T>> id_node_buf_map;
    auto poll_all = [&]() -> std::pair<int, int> {
      reader->poll_all(ctx);
      if (!on_flight_ios.empty())
        std::make_heap(on_flight_ios.begin(), on_flight_ios.end(), DecoupleIOCompare());
      unsigned n_in = 0, n_out = 0;
      while (!on_flight_ios.empty() && on_flight_ios.front().finished()) {
        decouple_io_t io = on_flight_ios.front();
        std::pop_heap(on_flight_ios.begin(), on_flight_ios.end(), DecoupleIOCompare());
        on_flight_ios.pop_back();
        reconcile_page_cache(io.read_req);
        id_node_buf_map.insert(std::make_pair(io.nbr.id, node_from_page((char *)io.read_req->buf, io.loc)));
        unlock_idx(idx_lock_table, io.nbr.id);
        ready_queue.push({io.nbr.id, io.nbr.distance});
        cached_pages.insert(io.page_id);
        unsigned bound = std::min<unsigned>(cur_list_size, (unsigned)l_search);
        io.nbr.distance <= retset[bound - 1].distance ? ++n_in : ++n_out;
      }
      return std::make_pair(n_in, n_out);
    };

    auto send_best_read_req = [&](uint32_t n) -> bool {
      unsigned n_sent = 0, marker = 0;
      const unsigned frontier = std::min<unsigned>(cur_list_size, (unsigned)l_search);
      while (marker < frontier && n_sent < n) {
        while (marker < frontier &&
               (retset[marker].flag == false || id_node_buf_map.find(retset[marker].id) != id_node_buf_map.end())) {
          retset[marker].flag = false;
          ++marker;
        }
        if (marker >= frontier) break;
        n_sent += send_read_req(retset[marker], marker);
      }
      return n_sent != 0;
    };
#else
    std::queue<decouple_io_t> on_flight_ios;
    auto send_read_req = [&](Neighbor &item) -> bool {
      item.flag = false;
      const unsigned loc = id2loc(item.id);
      const unsigned pid = loc_sector_no(loc);
      lock_idx(idx_lock_table, kInvalidID, std::vector<uint32_t>{item.id}, true);
      auto it = page_io_map.find(pid);
      if (it != page_io_map.end()) {
        on_flight_ios.push({item, pid, loc, it->second.req});
#ifndef READ_ONLY_TESTS
        page_ref->push_back(pid);
#endif
        return true;
      }
      uint64_t &cur_buf_idx = query_buf->sector_idx;
      auto buf = sector_scratch + cur_buf_idx * size_per_io;
      auto &req = query_buf->reqs[cur_buf_idx];
      req = IORequest(static_cast<uint64_t>(pid) * SECTOR_LEN, size_per_io, buf,
                      u_loc_offset(loc), meta_.max_node_len, sector_scratch);
      reader->send_read_no_alloc(req, ctx);
#ifndef READ_ONLY_TESTS
      page_ref->push_back(pid);
#endif
      page_io_map[pid] = {&req, 0};
      on_flight_ios.push({item, pid, loc, &req});
      cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;
      if (stats != nullptr) stats->n_ios++;
      return true;
    };

    std::unordered_map<unsigned, DiskNode<T>> id_node_buf_map;
    auto poll_all = [&]() -> std::pair<int, int> {
      reader->poll_all(ctx);
      unsigned n_in = 0, n_out = 0;
      while (!on_flight_ios.empty() && on_flight_ios.front().finished()) {
        decouple_io_t &io = on_flight_ios.front();
        reconcile_page_cache(io.read_req);
        id_node_buf_map.insert(std::make_pair(io.nbr.id, node_from_page((char *)io.read_req->buf, io.loc)));
        unlock_idx(idx_lock_table, io.nbr.id);
        ready_queue.push({io.nbr.id, io.nbr.distance});
        cached_pages.insert(io.page_id);
        unsigned bound = std::min<unsigned>(cur_list_size, (unsigned)l_search);
        io.nbr.distance <= retset[bound - 1].distance ? ++n_in : ++n_out;
        on_flight_ios.pop();
      }
      return std::make_pair(n_in, n_out);
    };

    auto send_best_read_req = [&](uint32_t n) -> bool {
      unsigned n_sent = 0, marker = 0;
      const unsigned frontier = std::min<unsigned>(cur_list_size, (unsigned)l_search);
      while (marker < frontier && n_sent < n) {
        while (marker < frontier &&
               (retset[marker].flag == false || id_node_buf_map.find(retset[marker].id) != id_node_buf_map.end())) {
          retset[marker].flag = false;
          ++marker;
        }
        if (marker >= frontier) break;
        n_sent += send_read_req(retset[marker]);
      }
      return n_sent != 0;
    };
#endif

    auto activate_node = [&](DiskNode<T> &node, unsigned loc, int cache_pos = -1) -> bool {
      unsigned id = loc2id(loc);
      if (expanded_set.find(id) != expanded_set.end()) return false;
      if (cache_pos != -1) {
        retset[cache_pos].flag = false;
        retset[cache_pos].visited = true;
      } else {
        for (unsigned i = 0; i < cur_list_size; ++i) {
          if (retset[i].id == id) { retset[i].flag = false; retset[i].visited = true; break; }
        }
      }
      expanded_set.insert(id);
      compute_exact_and_record(node, id);
      unsigned nk = cur_list_size;
      compute_and_push_nbrs(node, nk);
      return true;
    };

    auto calc_cache = [&]() -> bool {
      for (unsigned i = 0; i < cur_list_size; ++i) {
        if (retset[i].visited) continue;
        const uint32_t id = retset[i].id;
        unsigned loc = id2loc(id);
        unsigned pid = loc_sector_no(loc);
        if (cached_pages.find(pid) == cached_pages.end()) continue;
        auto locked = lock_idx(idx_lock_table, kInvalidID, std::vector<uint32_t>{id}, true);
        loc = id2loc(id); pid = loc_sector_no(loc);
        if (cached_pages.find(pid) == cached_pages.end()) { unlock_idx(idx_lock_table, locked); continue; }
        auto info_it = page_io_map.find(pid);
        if (info_it == page_io_map.end()) { unlock_idx(idx_lock_table, locked); continue; }
        auto map_it = id_node_buf_map.find(id);
        if (map_it == id_node_buf_map.end()) {
          DiskNode<T> n = node_from_page((char *)info_it->second.req->buf, loc);
          map_it = id_node_buf_map.insert(std::make_pair(id, n)).first;
        }
        bool ok = activate_node(map_it->second, loc, i);
        unlock_idx(idx_lock_table, locked);
        if (ok) return true;
      }
      return false;
    };

    auto expand_one = [&](unsigned id, float distance) -> bool {
      if (expanded_set.find(id) != expanded_set.end()) return false;
      for (unsigned i = 0; i < cur_list_size; ++i) {
        if (retset[i].id == id && !retset[i].visited) {
          retset[i].flag = false; retset[i].visited = true;
          expanded_set.insert(id);
          auto it = id_node_buf_map.find(id);
          if (it == id_node_buf_map.end()) return false;
          compute_exact_and_record(it->second, id);
          unsigned nk = cur_list_size;
          compute_and_push_nbrs(it->second, nk);
          return true;
        }
      }
      return false;
    };

    auto find_first_unvisited_eager = [&]() -> unsigned {
      const unsigned frontier = std::min<unsigned>(cur_list_size, (unsigned)l_search);
      for (unsigned i = 0; i < frontier; ++i)
        if (!retset[i].visited && retset[i].flag && id_node_buf_map.find(retset[i].id) == id_node_buf_map.end())
          return i;
      return frontier;
    };

    auto get_first_unvisited = [&]() -> int {
      const unsigned frontier = std::min<unsigned>(cur_list_size, (unsigned)l_search);
      for (unsigned i = 0; i < frontier; ++i)
        if (!retset[i].visited) return i;
      return -1;
    };

    auto cpu2_st = std::chrono::high_resolution_clock::now();
    send_best_read_req(cur_beam_width - on_flight_ios.size());
    unsigned max_marker = 0;
    int cur_n_in = 0, cur_tot = 0;

    while (get_first_unvisited() != -1) {
      Timer io_timer;
      auto [n_in, n_out] = poll_all();
      if (stats != nullptr) {
        stats->io_us += (double) io_timer.elapsed();
      }
      if (max_marker >= 5 && n_in + n_out > 0) {
        cur_n_in += n_in; cur_tot += n_in + n_out;
        constexpr double kWasteThreshold = 0.1;
        if ((cur_tot - cur_n_in) * 1.0 / cur_tot <= kWasteThreshold) {
          cur_beam_width = std::max(4l, std::min((int64_t)beam_width, cur_beam_width + 1));
        }
      }
      if (!ready_queue.empty()) {
        auto e = ready_queue.front(); ready_queue.pop();
        expand_one(e.id, e.distance);
      } else if (max_marker < 10) {
#ifdef CACHE_N
        calc_cache();
#endif
      } else if ((int64_t)on_flight_ios.size() == cur_beam_width) {
#ifdef CACHE_N
        calc_cache();
#endif
      }
      io_timer.reset();
      if ((int64_t)on_flight_ios.size() < cur_beam_width) send_best_read_req(1);
      if (stats != nullptr) {
        stats->io_us += (double) io_timer.elapsed();
      }
      if (max_marker < 5) max_marker = std::max(max_marker, find_first_unvisited_eager());
    }

    // Drain remaining on-flight IOs
    {
      while (!on_flight_ios.empty()) {
        Timer io_timer;
        std::ignore = poll_all();
        if (stats != nullptr) {
          stats->io_us += (double) io_timer.elapsed();
        }
      }
      for (unsigned i = 0; i < cur_list_size; ++i)
        if (!retset[i].visited && id_node_buf_map.find(retset[i].id) != id_node_buf_map.end()) {
          retset[i].visited = true;
          compute_exact_and_record(id_node_buf_map.find(retset[i].id)->second, retset[i].id);
        }
    }
#ifndef READ_ONLY_TESTS
    if (passthrough_page_ref == nullptr) {
      reader->deref(page_ref, nullptr);
      delete page_ref;
    }
#endif

    // Sort by exact distance (already computed from coords — no Phase 2)
    auto rerank_st = std::chrono::high_resolution_clock::now();
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &a, const Neighbor &b) { return a < b; });

    if (full_retset.size() > max_candidates) full_retset.resize(max_candidates);

    // Populate coord_map from cached DiskNode coords
    if (coord_map != nullptr && coord_buf != nullptr) {
      const uint64_t coord_cap = static_cast<uint64_t>(this->params.L) * INSERT_COORD_BUF_L_FACTOR;
      for (size_t i = 0; i < full_retset.size(); ++i) {
        uint32_t id = full_retset[i].id;
        if (coord_map->find(id) != coord_map->end()) continue;
        auto it = id_node_buf_map.find(id);
        if (it != id_node_buf_map.end()) {
          if (unlikely(i >= coord_cap)) {
            LOG(ERROR) << "Please allocate larger coord_buf."; crash();
          }
          memcpy(coord_buf + i * aligned_dim, it->second.coords, meta_.data_dim * sizeof(T));
          coord_map->insert(std::make_pair(id, coord_buf + i * aligned_dim));
        }
      }
    }

    expanded_nodes_info.assign(full_retset.begin(), full_retset.end());

    auto cpu2_ed = std::chrono::high_resolution_clock::now();
    if (stats != nullptr) {
      stats->cpu_us2 = std::chrono::duration_cast<std::chrono::microseconds>(cpu2_ed - cpu2_st).count();
      stats->cpu_us = n_computes;
      stats->rerank_us = std::chrono::duration_cast<std::chrono::microseconds>(cpu2_ed - rerank_st).count();
    }

    uint64_t t = 0;
    if (res_tags != nullptr) {
      for (uint64_t i = 0; i < full_retset.size() && t < k_search; i++) {
        res_tags[t] = id2tag(full_retset[i].id);
        if (distances != nullptr) distances[t] = full_retset[i].distance;
        t++;
      }
    }

    push_query_buf(query_buf);
    if (stats != nullptr) stats->total_us = (double)query_timer.elapsed();
    return t;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
