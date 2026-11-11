#include "aligned_file_reader.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include "utils/timer.h"
#include "utils/tsl/robin_map.h"
#include "utils/tsl/robin_set.h"
#include "utils.h"
#include "utils/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

namespace pipeann {
  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::page_search(const T *query1, const uint64_t k_search, const uint32_t mem_L,
                                        const uint64_t l_search, TagT *res_tags, float *distances,
                                        const uint64_t beam_width, QueryStats *stats) {
    QueryBuffer<T> *query_buf = pop_query_buf(query1);
    void *ctx = reader->get_ctx();

    if (beam_width > MAX_N_SECTOR_READS) {
      LOG(ERROR) << "Beamwidth can not be higher than MAX_N_SECTOR_READS";
      crash();
    }
    const T *query = query_buf->aligned_query_T;

    // reset query
    query_buf->reset();

    // pointers to current vector for comparison
    T *data_buf = query_buf->coord_scratch;
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    // sector scratch
    char *sector_scratch = query_buf->sector_scratch;
    uint64_t &sector_scratch_idx = query_buf->sector_idx;

    // query <-> PQ chunk centers distances
    nbr_handler->initialize_query(query, query_buf);
    float *dist_scratch = query_buf->aligned_dist_scratch;

    Timer query_timer, io_timer, cpu_timer;
    std::vector<Neighbor> retset(4096);
    tsl::robin_set<uint64_t> &visited = *(query_buf->visited);
    tsl::robin_set<unsigned> &page_visited = *(query_buf->page_visited);
    unsigned cur_list_size = 0;

    std::vector<Neighbor> full_retset;
    full_retset.reserve(4096);

    auto compute_exact_dists_and_push = [&](const DiskNode<T> &node, const unsigned id) -> float {
      T *node_fp_coords_copy = data_buf;
      memcpy(node_fp_coords_copy, node.coords, meta_.data_dim * sizeof(T));
      float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
      full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
      return cur_expanded_dist;
    };

    auto compute_and_push_nbrs = [&](DiskNode<T> &node, unsigned &nk) {
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
          const float nbor_dist = query_buf->aligned_dist_scratch[m];
          if (stats != nullptr) {
            stats->n_cmps++;
          }
          if (nbor_dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search))
            continue;
          Neighbor nn(nbor_id, nbor_dist, true);
          // Return position in sorted list where nn inserted
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);  // may be overflow in retset...
          if (cur_list_size < l_search)
            ++cur_list_size;
          // nk logs the best position in the retset that was updated due to
          // neighbors of n.
          if (r < nk)
            nk = r;
        }
      }
    };

    auto compute_and_add_to_retset = [&](const unsigned *node_ids, const uint64_t n_ids) {
      nbr_handler->compute_dists(query_buf, node_ids, n_ids);
      for (uint64_t i = 0; i < n_ids; ++i) {
        retset[cur_list_size].id = node_ids[i];
        retset[cur_list_size].distance = query_buf->aligned_dist_scratch[i];
        retset[cur_list_size++].flag = true;
        visited.insert(node_ids[i]);
      }
    };

    // stats.
    stats->io_us = 0;
    stats->cpu_us = 0;
    // search in in-memory index.
    if (mem_L) {
      std::vector<unsigned> mem_tags(mem_L);
      std::vector<float> mem_dists(mem_L);
      mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data());
      std::vector<uint32_t> live_ids(mem_L);
      const uint32_t n_live = resolve_mem_tags_to_ids(mem_tags.data(), mem_L, live_ids.data());
      if (n_live > 0) {
        compute_and_add_to_retset(live_ids.data(), std::min((unsigned) n_live, (unsigned) l_search));
      } else {
        compute_and_add_to_retset(&meta_.entry_point_id, 1);
      }
    } else {
      compute_and_add_to_retset(&meta_.entry_point_id, 1);
    }

    std::sort(retset.begin(), retset.begin() + cur_list_size);

    unsigned num_ios = 0;
    unsigned k = 0;

    // cleared every iteration
    std::vector<unsigned> frontier;
    frontier.reserve(2 * beam_width);
    using page_fnhood_t = std::tuple<unsigned, unsigned, PageArr, char *>;  // <node_id, page_id, page_layout, page_buf>
    std::vector<page_fnhood_t> frontier_nhoods;
    frontier_nhoods.reserve(2 * beam_width);
    std::vector<IORequest> frontier_read_reqs;
    frontier_read_reqs.reserve(2 * beam_width);

    using io_ss_t = std::tuple<unsigned, unsigned, PageArr>;  // <node_id, page_id, page_layout>
    std::vector<io_ss_t> last_io_snapshot;
    last_io_snapshot.reserve(2 * beam_width);

    std::vector<char> last_pages(size_per_io * beam_width * 2);

    // Logical nodes per disk page (0 means one multi-sector node per read).
    const unsigned nodes_per_page = meta_.nnodes_per_sector == 0 ? 1u : (unsigned) meta_.nnodes_per_sector;

    // search on disk.
    while (k < cur_list_size) {
      unsigned nk = cur_list_size;
      // clear iteration state
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      sector_scratch_idx = 0;
      // find new beam
      uint32_t marker = k;
      uint32_t num_seen = 0;

      // distribute cache and disk-read nodes
      // 100 us
      while (marker < cur_list_size && frontier.size() < beam_width && num_seen < beam_width) {
        const unsigned pid = id2page(retset[marker].id);
        if (page_visited.find(pid) == page_visited.end() && retset[marker].flag) {
          num_seen++;
          // disable nhood cache.
          frontier.push_back(retset[marker].id);
          page_visited.insert(pid);
          retset[marker].flag = false;
        }
        marker++;
      }

      // read nhoods of frontier ids
      std::vector<uint32_t> locked, page_locked;
      int n_ios = 0;
      if (!frontier.empty()) {
        if (stats != nullptr)
          stats->n_hops++;

        locked = this->lock_idx(idx_lock_table, kInvalidID, frontier, true);
        page_locked = this->lock_page_idx(page_idx_lock_table, kInvalidID, frontier, true);

        for (uint64_t i = 0; i < frontier.size(); i++) {
          auto id = frontier[i];
          unsigned loc = id2loc(id);
          uint64_t page_id = id2page(id);
          auto buf = sector_scratch + sector_scratch_idx * size_per_io;
          PageArr layout = get_page_layout(page_id);
          page_fnhood_t fnhood = std::make_tuple(id, page_id, layout, buf);
          sector_scratch_idx++;
          frontier_nhoods.push_back(fnhood);
          if (meta_.nnodes_per_sector > 1) {
            // Starling page layout: one read covers all nodes on the page.
            frontier_read_reqs.emplace_back(
                IORequest(page_id * SECTOR_LEN, size_per_io, buf, page_id * SECTOR_LEN, size_per_io));
          } else {
            // One node per page (or multi-sector node): same IO as beam_search.
            frontier_read_reqs.emplace_back(
                IORequest(loc_sector_no(loc) * SECTOR_LEN, size_per_io, buf, u_loc_offset(loc), meta_.max_node_len,
                          sector_scratch));
          }
          if (stats != nullptr) {
            stats->n_4k++;
            stats->n_ios++;
          }
          num_ios++;
        }

        n_ios = reader->send_read_no_alloc(frontier_read_reqs, ctx);
      }

      // compute remaining nodes in the pages that are fetched in the previous
      // round
      auto cpu1_st = std::chrono::high_resolution_clock::now();
      for (size_t i = 0; i < last_io_snapshot.size(); ++i) {
        auto &[last_io_id, pid, page_layout] = last_io_snapshot[i];
        char *sector_buf = last_pages.data() + i * size_per_io;

        // minus one for the vector that is computed previously
        std::vector<std::pair<float, DiskNode<T>>> vis_cand;  // <dist, node>
        vis_cand.reserve(nodes_per_page);

        // compute exact distances of the vectors within the page
        for (unsigned j = 0; j < nodes_per_page; ++j) {
          const unsigned id = page_layout[j];
          if (id == last_io_id || id == kAllocatedID || id == kInvalidID) {
            continue;
          }
          DiskNode<T> node = node_from_page(sector_buf, id2loc(id));
          float dist = compute_exact_dists_and_push(node, id);
          vis_cand.emplace_back(dist, node);
        }

        // compute PQ distances for neighbours of the vectors in the page
        for (unsigned k = 0; k < vis_cand.size(); ++k) {
          compute_and_push_nbrs(vis_cand[k].second, nk);
        }
      }
      last_io_snapshot.clear();
      auto cpu1_ed = std::chrono::high_resolution_clock::now();
      stats->cpu_us1 += std::chrono::duration_cast<std::chrono::microseconds>(cpu1_ed - cpu1_st).count();

      auto io_time_st = std::chrono::high_resolution_clock::now();
      // get last submitted io results, blocking
      if (!frontier.empty()) {
        for (int i = 0; i < n_ios; ++i) {
          reader->poll_wait(ctx);
        }
        this->unlock_page_idx(page_idx_lock_table, page_locked);
        this->unlock_idx(idx_lock_table, locked);
      }
      auto io_time_ed = std::chrono::high_resolution_clock::now();
      stats->io_us += std::chrono::duration_cast<std::chrono::microseconds>(io_time_ed - io_time_st).count();

      auto cpu_st = std::chrono::high_resolution_clock::now();
      // compute only the desired vectors in the pages - one for each page
      // postpone remaining vectors to the next round
      for (auto &[id, pid, layout, sector_buf] : frontier_nhoods) {
        // fill in the last_io_ids() and last_pages() with neighbor buffers.
        memcpy(last_pages.data() + last_io_snapshot.size() * size_per_io, sector_buf, size_per_io);
        last_io_snapshot.emplace_back(std::make_tuple(id, pid, layout));

        for (unsigned j = 0; j < nodes_per_page; ++j) {
          unsigned cur_id = layout[j];
          if (cur_id == id) {
            DiskNode<T> node = node_from_page(sector_buf, id2loc(id));
            compute_exact_dists_and_push(node, id);
            compute_and_push_nbrs(node, nk);
          }
        }
      }
      auto cpu_ed = std::chrono::high_resolution_clock::now();
      stats->cpu_us += std::chrono::duration_cast<std::chrono::microseconds>(cpu_ed - cpu_st).count();

      // update best inserted position
      if (nk <= k)
        k = nk;  // k is the best position in retset updated in this round.
      else
        ++k;
    }

    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    // copy k_search values
    uint64_t t = 0;
    for (uint64_t i = 0; i < full_retset.size() && t < k_search; i++) {
      if (i > 0 && full_retset[i].id == full_retset[i - 1].id) {
        continue;
      }
      res_tags[t] = id2tag(full_retset[i].id);
      if (distances != nullptr) {
        distances[t] = full_retset[i].distance;
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
