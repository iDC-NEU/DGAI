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
#include <limits>
#include <tuple>
#include "utils/timer.h"
#include "utils/tsl/robin_set.h"
#include "utils.h"
#include "utils/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

namespace pipeann {

  // Lightweight I/O tracking struct for vector reads in Phase 2.
  struct decouple_page_vec_io_t {
    IORequest *read_req;
    bool finished() const {
      return read_req->finished;
    }
  };

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::decouple_page_search(const T *query1, const uint64_t k_search, const uint32_t mem_L,
                                                  const uint64_t l_search, TagT *res_tags, float *distances,
                                                  const uint64_t beam_width, QueryStats *stats) {
    if (!decouple_mode_) {
      LOG(ERROR) << "decouple_page_search called but index is not in decouple mode";
      crash();
    }

    QueryBuffer<T> *query_buf = pop_query_buf(query1);
#ifdef USE_AIO
    void *ctx = graph_reader_->get_ctx();
#else
    void *ctx = graph_reader_->get_ctx(IORING_SETUP_SQPOLL);
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
    uint64_t &sector_scratch_idx = query_buf->sector_idx;

    nbr_handler->initialize_query(query, query_buf);
    float *dist_scratch = query_buf->aligned_dist_scratch;

    Timer query_timer, io_timer, cpu_timer;
    std::vector<Neighbor> retset(4096);
    tsl::robin_set<uint64_t> &visited = *(query_buf->visited);
    tsl::robin_set<unsigned> &page_visited = *(query_buf->page_visited);
    unsigned cur_list_size = 0;

    std::vector<Neighbor> full_retset;
    full_retset.reserve(4096);

    // Record a node (by ID) with its PQ distance from retset.
    auto record_node = [&](const unsigned id, const float pq_dist) {
      full_retset.push_back(Neighbor(id, pq_dist, true));
    };

    auto compute_and_push_nbrs = [&](DiskGraphNode<T> &node, unsigned &nk) {
      unsigned *node_nbrs = node.nbrs;
      unsigned nnbrs = node.nnbrs;
      unsigned nbors_cand_size = 0;
      for (unsigned m = 0; m < nnbrs; ++m) {
        if (visited.find(node_nbrs[m]) == visited.end()) {
          node_nbrs[nbors_cand_size++] = node_nbrs[m];
          visited.insert(node_nbrs[m]);
        }
      }
      if (nbors_cand_size) {
        nbr_handler->compute_dists(query_buf, node_nbrs, nbors_cand_size);
        for (unsigned m = 0; m < nbors_cand_size; ++m) {
          const int nbor_id = node_nbrs[m];
          const float nbor_dist = dist_scratch[m];
          if (stats != nullptr) {
            stats->n_cmps++;
          }
          if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search))
            continue;
          Neighbor nn(nbor_id, nbor_dist, true);
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
          if (cur_list_size < l_search)
            ++cur_list_size;
          if (r < nk)
            nk = r;
        }
      }
    };

    auto compute_and_add_to_retset = [&](const unsigned *node_ids, const uint64_t n_ids) {
      nbr_handler->compute_dists(query_buf, node_ids, n_ids);
      for (uint64_t i = 0; i < n_ids; ++i) {
        retset[cur_list_size].id = node_ids[i];
        retset[cur_list_size].distance = dist_scratch[i];
        retset[cur_list_size++].flag = true;
        visited.insert(node_ids[i]);
      }
    };

    // stats.
    if (stats != nullptr) {
      stats->io_us = 0;
      stats->cpu_us = 0;
    }

    // search in in-memory index.
    if (mem_L) {
      std::vector<uint32_t> mem_tags(mem_L);
      const uint32_t n_mem =
          mem_search_with_pq(query, query_buf, mem_L, mem_tags.data(), dist_scratch);
      for (uint64_t i = 0; i < std::min((uint64_t) n_mem, std::min((uint64_t) mem_L, (uint64_t) l_search)); ++i) {
        retset[cur_list_size].id = mem_tags[i];
        retset[cur_list_size].distance = dist_scratch[i];
        retset[cur_list_size++].flag = true;
        visited.insert(mem_tags[i]);
      }
    } else {
      compute_and_add_to_retset(&meta_.entry_point_id, 1);
    }

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned num_ios = 0;
    unsigned k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    std::vector<float> frontier_dists;  // PQ distance per frontier node (saved before flag cleared)
    frontier.reserve(2 * beam_width);
    frontier_dists.reserve(2 * beam_width);
    // <node_id, page_id, sector_buf>
    using page_fnhood_t = std::tuple<unsigned, unsigned, char *>;
    std::vector<page_fnhood_t> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<IORequest> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);

    // Deferred page processing: graph nodes from previous round's pages
    // that were not the frontier node (same pattern as page_search).
    using io_ss_t = std::tuple<unsigned, unsigned, PageArr>;  // <frontier_id, page_id, page_layout>
    std::vector<io_ss_t> last_io_snapshot;
    last_io_snapshot.reserve(2 * beam_width);
    // Use size_per_io_graph_ for graph page buffer (may span multiple sectors per node).
    std::vector<char> last_pages(size_per_io_graph_ * beam_width * 2);

    const unsigned nodes_per_page =
        nnodes_per_sector_graph_ == 0 ? 1u : (unsigned) nnodes_per_sector_graph_;
    const uint64_t graph_sectors_per_node =
        nnodes_per_sector_graph_ == 0 ? DIV_ROUND_UP(max_graph_node_len_, SECTOR_LEN) : 1;

    auto get_graph_page_layout = [&](uint32_t page_no) -> PageArr {
      PageArr ret;
      if (nnodes_per_sector_graph_ == 0) {
        // One graph node spans multiple sectors; page_no = loc * graph_sectors_per_node.
        const uint64_t loc = page_no / graph_sectors_per_node;
        ret.push_back(loc2id((uint32_t) loc));
      } else {
        const uint64_t st = (uint64_t) page_no * nnodes_per_sector_graph_;
        const uint64_t ed = st + nnodes_per_sector_graph_;
        for (uint64_t loc = st; loc < ed; ++loc) {
          ret.push_back(loc2id((uint32_t) loc));
        }
      }
      return ret;
    };

    // --- Phase 1: Page-based search on graph-only nodes (PQ distances only) ---
    while (k < cur_list_size) {
      unsigned nk = cur_list_size;
      // clear iteration state
      frontier.clear();
      frontier_dists.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      sector_scratch_idx = 0;

      // find new beam: select nodes on pages not yet visited
      uint32_t marker = k;
      uint32_t num_seen = 0;
      while (marker < cur_list_size && frontier.size() < beam_width && num_seen < beam_width) {
        unsigned loc = id2loc(retset[marker].id);
        const unsigned pid = graph_loc_sector_no(loc);
        if (page_visited.find(pid) == page_visited.end() && retset[marker].flag) {
          num_seen++;
          frontier.push_back(retset[marker].id);
          frontier_dists.push_back(retset[marker].distance);  // save PQ distance before clearing flag
          page_visited.insert(pid);
          retset[marker].flag = false;
        }
        marker++;
      }

      // read full graph pages of frontier ids
      std::vector<uint32_t> locked;
      int n_ios = 0;
      if (!frontier.empty()) {
        if (stats != nullptr)
          stats->n_hops++;

        locked = this->lock_idx(idx_lock_table, kInvalidID, frontier, true);

        for (uint64_t i = 0; i < frontier.size(); i++) {
          auto id = frontier[i];
          unsigned loc = id2loc(id);
          uint64_t page_id = graph_loc_sector_no(loc);
          auto buf = sector_scratch + sector_scratch_idx * size_per_io_graph_;
          page_fnhood_t fnhood = std::make_tuple(id, page_id, buf);
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          // read the full graph page
          frontier_read_reqs.emplace_back(
              IORequest(page_id * SECTOR_LEN, size_per_io_graph_, buf,
                        page_id * SECTOR_LEN, size_per_io_graph_));
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
          num_ios++;
        }

        n_ios = graph_reader_->send_read_no_alloc(frontier_read_reqs, ctx);
      }

      // Process deferred nodes from previous round's pages (expand all co-resident nodes).
      struct deferred_cand_t {
        uint32_t id;
        uint64_t slot_in_page;
        float pq_dist;
      };
      auto cpu1_st = std::chrono::high_resolution_clock::now();
      for (size_t i = 0; i < last_io_snapshot.size(); ++i) {
        auto &[frontier_id, pid, page_layout] = last_io_snapshot[i];
        char *sector_buf = last_pages.data() + i * size_per_io_graph_;

        if (nnodes_per_sector_graph_ == 0) {
          continue;  // one node per page; frontier already processed
        }

        std::vector<deferred_cand_t> deferred_cands;
        deferred_cands.reserve(nodes_per_page);
        for (unsigned j = 0; j < nodes_per_page; ++j) {
          const uint32_t id = page_layout[j];
          if (id == frontier_id || id == kAllocatedID || id == kInvalidID) {
            continue;
          }
          deferred_cands.push_back({id, j, 0.f});
        }
        if (deferred_cands.empty()) {
          continue;
        }

        std::vector<unsigned> deferred_ids(deferred_cands.size());
        for (size_t j = 0; j < deferred_cands.size(); ++j) {
          deferred_ids[j] = deferred_cands[j].id;
        }
        nbr_handler->compute_dists(query_buf, deferred_ids.data(), deferred_ids.size());
        for (size_t j = 0; j < deferred_cands.size(); ++j) {
          deferred_cands[j].pq_dist = dist_scratch[j];
        }

        for (auto &cand : deferred_cands) {
          visited.insert(cand.id);
          const unsigned loc = id2loc(cand.id);
          DiskGraphNode<T> node = graph_node_from_page(sector_buf, loc);
          record_node(cand.id, cand.pq_dist);
          compute_and_push_nbrs(node, nk);
        }
      }
      last_io_snapshot.clear();
      auto cpu1_ed = std::chrono::high_resolution_clock::now();
      if (stats != nullptr) {
        stats->cpu_us1 += std::chrono::duration_cast<std::chrono::microseconds>(cpu1_ed - cpu1_st).count();
      }

      // Wait for current round's I/O to complete.
      auto io_time_st = std::chrono::high_resolution_clock::now();
      if (!frontier.empty()) {
        for (int i = 0; i < n_ios; ++i) {
          graph_reader_->poll_wait(ctx);
        }
        this->unlock_idx(idx_lock_table, locked);
      }
      auto io_time_ed = std::chrono::high_resolution_clock::now();
      if (stats != nullptr) {
        stats->io_us += std::chrono::duration_cast<std::chrono::microseconds>(io_time_ed - io_time_st).count();
      }

      // Process frontier nodes on the current round's pages.
      auto cpu_st = std::chrono::high_resolution_clock::now();
      for (uint64_t fi = 0; fi < frontier_nhoods.size(); fi++) {
        auto &[id, pid, sector_buf] = frontier_nhoods[fi];
        PageArr layout = get_graph_page_layout(pid);
        memcpy(last_pages.data() + last_io_snapshot.size() * size_per_io_graph_, sector_buf, size_per_io_graph_);
        last_io_snapshot.emplace_back(std::make_tuple(id, pid, layout));

        for (unsigned j = 0; j < nodes_per_page; ++j) {
          if (layout[j] != id) {
            continue;
          }
          const unsigned loc = id2loc(id);
          DiskGraphNode<T> node = graph_node_from_page(sector_buf, loc);
          record_node(id, frontier_dists[fi]);
          compute_and_push_nbrs(node, nk);
        }
      }
      auto cpu_ed = std::chrono::high_resolution_clock::now();
      if (stats != nullptr) {
        stats->cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(cpu_ed - cpu_st).count();
      }

      // update best inserted position
      if (nk <= k)
        k = nk;
      else
        ++k;
    }

    // Sort by PQ distance
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    // Deduplicate
    std::vector<Neighbor> unique_candidates;
    unique_candidates.reserve(full_retset.size());
    for (size_t i = 0; i < full_retset.size(); ++i) {
      if (i == 0 || full_retset[i].id != full_retset[i - 1].id) {
        unique_candidates.push_back(full_retset[i]);
      }
    }
    unique_candidates.resize(std::min(unique_candidates.size() * 1ULL, 30ULL));

    // --- Phase 2: Read exact vectors and compute exact distances for reranking ---
    uint64_t vec_data_len = meta_.data_dim * sizeof(T);
#ifdef USE_AIO
    void *vec_ctx = vector_reader_->get_ctx();
#else
    void *vec_ctx = vector_reader_->get_ctx();
#endif

    uint64_t vec_buf_idx = 0;

    for (size_t batch_start = 0; batch_start < unique_candidates.size(); batch_start += MAX_N_SECTOR_READS) {
      size_t batch_end = std::min(batch_start + MAX_N_SECTOR_READS, unique_candidates.size());
      size_t batch_size = batch_end - batch_start;

      std::vector<decouple_page_vec_io_t> vec_ios;
      vec_ios.reserve(batch_size);

      for (size_t i = batch_start; i < batch_end; ++i) {
        uint32_t id = unique_candidates[i].id;
        uint64_t sector_no = vec_loc_sector_no(id);
        uint64_t u_off = vec_sector_offset(id);

        uint64_t slot = vec_buf_idx % MAX_N_SECTOR_READS;
        char *buf = sector_scratch + slot * size_per_io_vec_;
        auto &req = query_buf->reqs[slot];
        req = IORequest(sector_no * SECTOR_LEN, size_per_io_vec_, buf,
                        u_off, vec_data_len, sector_scratch);
        // Resolve still-staged vectors from the in-memory overlay; only disk-read the
        // misses (which are guaranteed to be on disk). The compute loop below reads
        // req.buf + req.u_offset, which now holds the overlay copy for hits.
        if (!try_read_decouple_vector(id, buf + u_off)) {
          vector_reader_->send_read_no_alloc(req, vec_ctx);
          vec_ios.push_back({&req});
        }
        vec_buf_idx++;
      }

      Timer io_timer;
      while (true) {
        vector_reader_->poll_all(vec_ctx);
        bool all_done = true;
        for (auto &io : vec_ios) {
          if (!io.finished()) {
            all_done = false;
            break;
          }
        }
        if (all_done) break;
      }
      if (stats != nullptr) {
        stats->io_us += (double) io_timer.elapsed();
      }

      for (size_t i = 0; i < batch_size; ++i) {
        uint32_t id = unique_candidates[batch_start + i].id;
        auto &req = query_buf->reqs[(vec_buf_idx - batch_size + i) % MAX_N_SECTOR_READS];
        const T *raw_vec = (const T *) ((char *) req.buf + req.u_offset);
        float exact_dist = dist_cmp->compare(query, raw_vec, (unsigned) aligned_dim);
        unique_candidates[batch_start + i].distance = exact_dist;
      }
    }

    // Sort by exact distance
    std::sort(unique_candidates.begin(), unique_candidates.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    // Copy top k_search results
    uint64_t t = 0;
    for (uint64_t i = 0; i < unique_candidates.size() && t < k_search; i++) {
      res_tags[t] = id2tag(unique_candidates[i].id);
      if (distances != nullptr) {
        distances[t] = unique_candidates[i].distance;
      }
      t++;
    }

    push_query_buf(query_buf);

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
    return t;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
