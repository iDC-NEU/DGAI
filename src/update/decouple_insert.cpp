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
#include "utils.h"
#include "utils/page_cache.h"
#include "utils/prune_neighbors.h"
#include "utils/locality_split.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

namespace pipeann {
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::plan_pq_graph_split(
      uint32_t target_id, uint64_t source_page, const std::vector<uint32_t> &new_nhood,
      const std::unordered_map<uint64_t, char *> &page_buffers, QueryBuffer<T> *scratch,
      std::vector<uint64_t> &locs, std::vector<uint64_t> &split_orig_locs,
      std::vector<uint64_t> &split_new_locs, std::vector<uint32_t> &split_ids,
      OnlineSplitEvent &event) {
    Timer plan_timer;
    // Restrict only the actual decoupled split path. Constructing an index,
    // coupled operations, and read-only searches need no PQ/L2 restriction.
    if (metric != pipeann::Metric::L2 || dynamic_cast<PQNeighbor<T> *>(nbr_handler) == nullptr) {
      LOG(ERROR) << "Decoupled topology split requires the PQNeighbor L2 backend";
      crash();
    }
    if (source_page == kInvalidID || locs.size() != new_nhood.size() + 1 ||
        !split_orig_locs.empty() || !split_new_locs.empty() || !split_ids.empty()) {
      LOG(ERROR) << "Invalid topology split reservation";
      crash();
    }
    const uint64_t capacity = nnodes_per_sector_graph_;
    const uint64_t old_page = source_page;
    const uint64_t new_page = graph_loc_sector_no(locs.back());
    const uint64_t old_start = old_page * capacity, new_start = new_page * capacity;
    if (old_page == new_page || page_buffers.count(old_page) == 0 || page_buffers.count(new_page) == 0) {
      LOG(ERROR) << "PQ split requires distinct, already-read source and reserved destination buffers";
      crash();
    }

    Timer snapshot_timer;
    std::vector<uint32_t> ids;
    std::vector<uint64_t> old_locs;
    ids.reserve(capacity + 1);
    old_locs.reserve(capacity);
    {
      Timer wait_timer;
      std::lock_guard<std::mutex> guard(alloc_lock);
      event.plan_wait_us += wait_timer.elapsed();
      for (uint64_t slot = 0; slot < capacity; ++slot) {
        const uint64_t loc = old_start + slot;
        const uint32_t id = loc2id(loc);
        if (id == kInvalidID || id == kAllocatedID) continue;
        if (id == target_id || id2loc(id) != loc) {
          LOG(ERROR) << "Unstable source mapping during PQ split: id=" << id;
          crash();
        }
        ids.push_back(id);
        old_locs.push_back(loc);
      }
    }
    const size_t old_count = ids.size();
    if (old_count < 2 || old_count > MAX_N_EDGES) {
      LOG(ERROR) << "Unsupported PQ split source count: " << old_count;
      crash();
    }
    ids.push_back(target_id);
    const size_t n = ids.size();
    std::vector<std::vector<uint32_t>> adjacency(n);
    for (size_t i = 0; i < old_count; ++i) {
      const auto node = graph_node_from_page(page_buffers.at(old_page), old_locs[i], ids[i],
                                              "insert/pq_split/source");
      if (node.nnbrs > meta_.range) {
        LOG(ERROR) << "Invalid adjacency size during PQ split: id=" << ids[i] << " degree=" << node.nnbrs;
        crash();
      }
      adjacency[i].assign(node.nbrs, node.nbrs + node.nnbrs);
    }
    // This is the available insertion-time topology snapshot. The new node's
    // pruned outgoing list is known; do not invent reciprocal edges that the
    // subsequent neighbor pruning may discard.
    adjacency.back() = new_nhood;
    event.topology_snapshot_us += snapshot_timer.elapsed();

    // New/reused IDs have already been encoded by nbr_handler->insert at the
    // beginning of this insertion. No raw vectors or additional pages are read.
    Timer pq_timer;
    std::vector<float> pq_distances(n * n, 0.0f);
    for (size_t i = 0; i + 1 < n; ++i) {
      const size_t count = n - i - 1;
      nbr_handler->compute_dists(ids[i], ids.data() + i + 1, count,
                                 scratch->aligned_dist_scratch, scratch->nbr_vec_scratch);
      ++event.pq_batches;
      event.pq_pairs += count;
      for (size_t offset = 0; offset < count; ++offset) {
        const size_t j = i + 1 + offset;
        pq_distances[i * n + j] = pq_distances[j * n + i] = scratch->aligned_dist_scratch[offset];
      }
    }
    event.pq_distance_us += pq_timer.elapsed();

    Timer partition_timer;
    std::unordered_map<uint32_t, size_t> local_index;
    local_index.reserve(n);
    for (size_t i = 0; i < n; ++i) local_index.emplace(ids[i], i);
    const auto partition = locality_split::weighted_bisection(
        ids, adjacency, capacity, [&](uint32_t a, uint32_t b) {
          return pq_distances[local_index.at(a) * n + local_index.at(b)];
        });
    event.partition_us += partition_timer.elapsed();
    const auto &diag = partition.diagnostics;
    event.local_directed_edges = diag.directed_edges;
    event.zero_degree_nodes = diag.zero_degree_nodes;
    event.no_edge_fallbacks = diag.no_edge_fallback;
    event.initializations = diag.initializations;
    event.sweeps = diag.sweeps;
    event.refinement_moves = diag.accepted_moves;

    size_t old_members[2] = {0, 0};
    for (size_t i = 0; i < old_count; ++i) ++old_members[partition.group[i]];
    const uint8_t stay = old_members[1] > old_members[0] ? 1 : 0;
    const bool target_stays = partition.group.back() == stay;
    std::vector<uint64_t> planned_old_locs, planned_new_locs;
    std::vector<uint32_t> planned_ids;
    for (size_t i = 0; i < old_count; ++i) {
      if (partition.group[i] == stay) continue;
      planned_old_locs.push_back(old_locs[i]);
      planned_new_locs.push_back(new_start + planned_ids.size());
      planned_ids.push_back(ids[i]);
    }
    const size_t moved = planned_ids.size();
    const size_t new_size = moved + !target_stays;
    const size_t old_size = old_count - moved + target_stays;
    if (old_size == 0 || new_size == 0 || old_size > capacity || new_size > capacity ||
        (target_stays && planned_old_locs.empty())) {
      LOG(ERROR) << "Invalid PQ split group sizes: " << old_size << "/" << new_size;
      crash();
    }
    const uint64_t target_loc = target_stays ? planned_old_locs.front() : new_start + moved;

    {
      Timer wait_timer;
      std::lock_guard<std::mutex> guard(alloc_lock);
      event.plan_wait_us += wait_timer.elapsed();
      // No relocation has been published. Both page locks remain held while
      // the destination marker is replaced with the actual slot reservations.
      for (size_t i = 0; i < old_count; ++i) {
        if (loc2id(old_locs[i]) != ids[i] || id2loc(ids[i]) != old_locs[i]) {
          LOG(ERROR) << "Source mapping changed while its split page was locked";
          crash();
        }
      }
      for (uint64_t slot = 0; slot < capacity; ++slot) {
        const uint64_t loc = new_start + slot;
        const uint32_t owner = loc2id(loc);
        if (owner != kInvalidID && owner != kAllocatedID) {
          LOG(ERROR) << "Reserved split destination unexpectedly contains a live node";
          crash();
        }
        set_loc2id(loc, kInvalidID);
      }
      for (uint64_t loc : planned_new_locs) set_loc2id(loc, kAllocatedID);
      if (!target_stays) set_loc2id(target_loc, kAllocatedID);
      locs.back() = target_loc;
      split_orig_locs = std::move(planned_old_locs);
      split_new_locs = std::move(planned_new_locs);
      split_ids = std::move(planned_ids);
    }
    event.old_page_size = old_size;
    event.new_page_size = new_size;
    event.moved_old_nodes = moved;
    event.target_in_new_page = !target_stays;
    event.plan_us += plan_timer.elapsed();
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::decouple_insert_in_place(const T *point1, const TagT &tag,
                                                   tsl::robin_set<uint32_t> *deletion_set,
                                                   InsertStats *stats,
                                                   int search_mode) {
    Timer insert_timer;
    QueryBuffer<T> *read_data = this->pop_query_buf(point1);
    T *point = read_data->aligned_query_T;
    void *ctx = graph_reader_->get_ctx();

    uint32_t target_id;
    bool reused_id = false;
    {
      std::lock_guard<std::mutex> guard(alloc_lock);
      if (!free_decouple_ids_.empty()) {
        target_id = free_decouple_ids_.back();
        free_decouple_ids_.pop_back();
        reused_id = true;
        reused_vector_slots_.fetch_add(1, std::memory_order_relaxed);
      } else {
        if(memnav_ && cur_id.load()>=memnav_->max_ids)throw std::overflow_error("MemNav ID budget before insert");
        const uint64_t next_id = cur_id++;
        if (unlikely(next_id >= kAllocatedID)) {
          LOG(ERROR) << "Internal node ID space exhausted";
          crash();
        }
        target_id = static_cast<uint32_t>(next_id);
      }
    }

    // write neighbor (e.g., PQ).
    nbr_handler->insert(point, target_id);

    {
      Timer vec_timer;
      stage_decouple_vector_write(target_id, point, reused_id);
      if (stats) stats->vec_write_us = vec_timer.elapsed();
    }

    std::vector<Neighbor> exp_node_info;
    InsertContext insert_ctx(INSERT_COORD_BUF_L_FACTOR * params.L, this->aligned_dim);
    std::vector<uint64_t> beam_page_ref{};
    std::vector<uint64_t> *page_ref = nullptr;
    {
      Timer search_timer;
      if (search_mode == DECOUPLE_SEARCH || search_mode == STARLING_DECOUPLE_SEARCH) {
        QueryStats search_stats;
        this->decouple_search(point1, params.L, memnav_?32:0, params.L, exp_node_info, params.beam_width, &search_stats, nullptr,
                              nullptr, params.L, &insert_ctx);
        if (stats) stats->search_rerank_us = search_stats.rerank_us;
        page_ref = &insert_ctx.page_ref;
      } else {
        this->do_decouple_beam_search(point1, memnav_?32:0, params.L, params.beam_width, exp_node_info, &insert_ctx.coord_map,
                                      insert_ctx.coord_buf, nullptr, deletion_set, false, &beam_page_ref);
        page_ref = &beam_page_ref;
      }
      if (stats) stats->search_us = search_timer.elapsed();
    }

    // With DGAI_STRICT_GRAPH_CHECKS, crash if search returns an id without a live
    // graph location (should not happen once search/insert invariants hold).
#ifdef DGAI_STRICT_GRAPH_CHECKS
    for (auto &nb : exp_node_info) {
      check_graph_node_mapping(nb.id, id2loc(nb.id), "decouple_insert/candidate");
    }
#endif

    std::vector<uint32_t> new_nhood;
    std::vector<Neighbor> placement_neighbors;
    pipeann::prune_neighbors(exp_node_info, new_nhood, params, metric,
                             [this, &insert_ctx](uint32_t a, uint32_t b) {
                               return this->dist_cmp->compare(insert_ctx.coord_map[a], insert_ctx.coord_map[b],
                                                              this->meta_.data_dim);
                             }, &placement_neighbors);

    // locs[new_nhood.size()] is the target, locs[0:new_nhood.size() - 1] are the neighbors.
    //
    // Concurrency-safe placement.
    // Neighbors are updated in place (their graph loc never changes); only a
    // `split` relocates existing nodes. The danger is a *concurrent* insert that
    // splits a page holding one of our neighbors: it would move the neighbor and
    // flip its id2loc while we still believe it lives at the old loc. To prevent
    // this we (1) snapshot each neighbor's loc and lock its page, (2) re-validate
    // that no neighbor moved while we were acquiring the locks (retry from
    // scratch if one did), and only then (3) run allocation. Holding the
    // neighbor (= split hint) page locks also serializes two inserts that would
    // split the same page, so double-splits cannot happen. Tail pages retain
    // ascending lock order. Reclaimed pages can have lower addresses, so the
    // allocator acquires their locks only with trywrlock (never waiting while
    // holding neighbor locks), and we retain that lock until writeback.
    //
    // Build with -DDGAI_DISABLE_GRAPH_SPLIT (CMake: DGAI_DISABLE_GRAPH_SPLIT=ON)
    // to skip page splits; alloc_graph_loc then never returns split=true and the
    // target is placed in an empty hint slot or at the graph file tail.
    std::set<uint64_t> pages_need_to_read;
    std::vector<uint64_t> split_orig_locs, split_new_locs, locs;
    std::vector<uint32_t> split_ids;
    std::vector<uint64_t> nbr_pages_locked;
    uint64_t recycled_page = kInvalidID;
    uint64_t split_source_page = kInvalidID;
    bool split = false;
    OnlineSplitEvent split_event;

    Timer alloc_timer;
    while (true) {
      std::set<uint64_t> nbr_page_set;
      std::vector<uint64_t> snap_locs(new_nhood.size());
      for (size_t i = 0; i < new_nhood.size(); ++i) {
        snap_locs[i] = id2loc(new_nhood[i]);
        nbr_page_set.insert(graph_loc_sector_no(snap_locs[i]));
      }
      std::vector<IORequest> nbr_lock_reqs;
      for (auto p : nbr_page_set) {
        nbr_lock_reqs.push_back(IORequest(p * SECTOR_LEN, size_per_io_graph_, nullptr, 0, 0));
      }
      Timer lock_acq_timer;
      nbr_pages_locked = pipeann::lockReqs(this->page_lock_table, nbr_lock_reqs);
      if (stats) stats->find_loc_lock_us += lock_acq_timer.elapsed();

      bool stable = true;
      for (size_t i = 0; i < new_nhood.size(); ++i) {
        if (id2loc(new_nhood[i]) != snap_locs[i]) {  // relocated by a concurrent split.
          stable = false;
          break;
        }
      }
      if (stable) {
        pages_need_to_read.clear();
        double alloc_wait = 0;
        Timer alloc_call_timer;
        split = this->alloc_graph_loc(target_id, new_nhood, placement_neighbors, pages_need_to_read, locs, split_source_page,
                                      &alloc_wait, &recycled_page);
        const double allocation_us = alloc_call_timer.elapsed();
        if (split) {
          split_event.allocator_us = allocation_us;
          split_event.allocator_wait_us = alloc_wait;
        }
        if (stats) {
          stats->find_loc_alloc_us += allocation_us;
          stats->find_loc_alloc_wait_us += alloc_wait;
        }
        break;
      }
      pipeann::unlockReqs(this->page_lock_table, nbr_pages_locked);
      if (stats) stats->find_loc_retries += 1;
    }
    if (stats) stats->find_graph_loc_us = alloc_timer.elapsed();

    uint64_t target_loc = locs[new_nhood.size()];

    // Lock fresh tail pages not covered by the neighbor locks. The allocator
    // already holds any reclaimed page lock; never acquire it a second time.
    std::set<uint64_t> nbr_sectors(nbr_pages_locked.begin(), nbr_pages_locked.end());
    std::set<uint64_t> new_sectors;  // fresh or reclaimed: initialize an empty page
    {
      auto consider = [&](uint64_t loc) {
        uint64_t s = graph_loc_sector_no(loc);
        if (nbr_sectors.find(s) == nbr_sectors.end()) new_sectors.insert(s);
      };
      consider(target_loc);
    }
    std::vector<IORequest> extra_lock_reqs;
    for (auto s : new_sectors) {
      if (s != recycled_page)
        extra_lock_reqs.push_back(IORequest(s * SECTOR_LEN, size_per_io_graph_, nullptr, 0, 0));
    }
    Timer lock_timer;
    std::vector<uint64_t> extra_pages_locked = pipeann::lockReqs(this->page_lock_table, extra_lock_reqs);
    if (recycled_page != kInvalidID) extra_pages_locked.push_back(recycled_page);
    if (stats) stats->lock_wait_us = lock_timer.elapsed();

    // Allocate per-page buffers. Existing (neighbor) pages are read from disk /
    // cache; freshly allocated pages are zeroed (reading past EOF is undefined).
    std::unordered_map<uint64_t, char *> page_buf_map;
    std::vector<char *> all_bufs;
    std::vector<IORequest> reads;
    auto ensure_buf = [&](uint64_t sector, bool is_new) {
      if (page_buf_map.count(sector)) return;
      char *buf = nullptr;
      alloc_aligned((void **) &buf, size_per_io_graph_, SECTOR_LEN);
      if (is_new) {
        memset(buf, 0, size_per_io_graph_);
      } else {
        reads.push_back(IORequest(sector * SECTOR_LEN, size_per_io_graph_, buf, 0, 0));
      }
      page_buf_map[sector] = buf;
      all_bufs.push_back(buf);
    };
    for (auto s : nbr_sectors) ensure_buf(s, /*is_new=*/false);
    for (auto s : new_sectors) ensure_buf(s, /*is_new=*/true);

    Timer read_timer;
    graph_reader_->read_alloc(reads, ctx, page_ref);
    if (stats) stats->read_topo_us = read_timer.elapsed();

    if (split) {
      this->plan_pq_graph_split(target_id, split_source_page, new_nhood, page_buf_map, read_data, locs,
                                split_orig_locs, split_new_locs, split_ids, split_event);
      target_loc = locs.back();
      // Include post-read planning in the existing insertion phase breakdown
      // as well as the detailed, explicitly scoped split counters.
      if (stats) {
        stats->find_loc_alloc_us += split_event.plan_us;
        stats->find_loc_alloc_wait_us += split_event.plan_wait_us;
        stats->find_graph_loc_us += split_event.plan_us;
      }
    }

    // Build the new graph topology in the page buffers. id2loc(target) is set
    // before wbc_write below; split victims' id2loc are flipped under the idx
    // lock after their data has been staged at the new loc.
    Timer build_timer;
    {
      Timer prune_timer;
      for (uint32_t i = 0; i < new_nhood.size(); ++i) {
        const uint64_t read_loc = id2loc(new_nhood[i]);
        uint64_t write_loc = read_loc;
        if (split) {
          for (uint32_t j = 0; j < split_ids.size(); ++j) {
            if (new_nhood[i] == split_ids[j]) {
              write_loc = split_new_locs[j];
              break;
            }
          }
        }

        const uint64_t read_sector = graph_loc_sector_no(read_loc);
        const uint64_t write_sector = graph_loc_sector_no(write_loc);
        if (page_buf_map.find(read_sector) == page_buf_map.end() ||
            page_buf_map.find(write_sector) == page_buf_map.end()) {
          LOG(ERROR) << new_nhood[i] << " read_sector=" << read_sector << " write_sector=" << write_sector
                     << " not found in page_buf_map";
          crash();
        }

        DiskGraphNode<T> r_nbr_node =
            graph_node_from_page(page_buf_map[read_sector], read_loc, new_nhood[i], "insert/nbr_read");
        if (unlikely(r_nbr_node.nnbrs > meta_.range)) {
          LOG(ERROR) << "Corrupt graph nnbrs: nbr_id=" << new_nhood[i] << " nnbrs=" << r_nbr_node.nnbrs
                     << " max_range=" << meta_.range;
          crash();
        }

        std::vector<uint32_t> nhood(r_nbr_node.nnbrs + 1);
        nhood.assign(r_nbr_node.nbrs, r_nbr_node.nbrs + r_nbr_node.nnbrs);
        nhood.emplace_back(target_id);

        if (nhood.size() > this->params.R) {
          auto &thread_pq_buf = read_data->nbr_vec_scratch;
          auto nbr = this->nbr_handler;
          pipeann::delta_prune_neighbors(
              nhood, new_nhood[i], target_id, params, metric,
              [nbr, &thread_pq_buf](uint32_t center, const uint32_t *ids, uint32_t n, float *dists_out) {
                nbr->compute_dists(center, ids, n, dists_out, thread_pq_buf);
              });
        }

        check_graph_nnbrs((uint32_t) nhood.size(), new_nhood[i], write_loc, "insert/nbr_after_prune");
        DiskGraphNode<T> w_nbr_node =
            graph_node_from_page(page_buf_map[write_sector], write_loc, new_nhood[i], "insert/nbr_write");
        w_nbr_node.nnbrs = (uint32_t) nhood.size();
        memcpy(w_nbr_node.nbrs, nhood.data(), w_nbr_node.nnbrs * sizeof(uint32_t));
      }
      if (stats) stats->prune_us = prune_timer.elapsed();

      if (split) {
        for (uint32_t j = 0; j < split_ids.size(); ++j) {
          const uint64_t old_loc = split_orig_locs[j];
          const uint64_t new_loc = split_new_locs[j];
          const uint32_t id = split_ids[j];
          const uint64_t old_sector = graph_loc_sector_no(old_loc);
          char *old_buf = page_buf_map.at(old_sector);

          bool is_nbr = false;
          for (auto nbr_id : new_nhood) {
            if (nbr_id == id) {
              is_nbr = true;
              break;
            }
          }
          if (!is_nbr) {
            const uint64_t new_sector = graph_loc_sector_no(new_loc);
            char *new_buf = page_buf_map.at(new_sector);
            DiskGraphNode<T> src_node = graph_node_from_page(old_buf, old_loc, id, "insert/split/src");
            DiskGraphNode<T> dst_node = graph_node_from_page(new_buf, new_loc, id, "insert/split/dst");
            check_graph_nnbrs(src_node.nnbrs, id, old_loc, "insert/split/src_memcpy");
            dst_node.nnbrs = src_node.nnbrs;
            memcpy(dst_node.nbrs, src_node.nbrs, src_node.nnbrs * sizeof(uint32_t));
          }
          // NOTE: we intentionally do NOT clear the victim's old slot. A searcher
          // holding the idx read-lock may still resolve this victim to its old
          // loc until we flip id2loc below, so the old slot must keep valid data.
          // The old loc becomes unreferenced once loc2id is erased and is reused
          // (fully overwritten) by a later allocation.
        }
      }

      // Write the target node LAST. In the split case the target reuses the
      // first victim's (now relocated) slot, so it must be written only after
      // every read above has copied the original data out of that slot.
      auto sector = graph_loc_sector_no(target_loc);
      DiskGraphNode<T> target_node =
          graph_node_from_page(page_buf_map[sector], target_loc, target_id, "insert/target_write");
      check_graph_nnbrs((uint32_t) new_nhood.size(), target_id, target_loc, "insert/target_write");
      target_node.nnbrs = (uint32_t) new_nhood.size();
      memcpy(target_node.nbrs, new_nhood.data(), new_nhood.size() * sizeof(uint32_t));
      tags.insert_or_assign(target_id, tag);
    }
    if (stats) stats->build_topo_us = build_timer.elapsed();

    // Stage writes. The target's page must become visible to searchers only
    // *after* id2loc(target) is set; otherwise a searcher that follows a freshly
    // added neighbor->target edge could resolve target to an unmapped loc. We
    // therefore set id2loc(target) before wbc_write and stage the target page
    // first (a single cache sector becomes visible atomically, so even when the
    // target shares a sector with a neighbor the mapping is already in place).
    std::vector<IORequest> writes;
    const uint64_t target_sector = graph_loc_sector_no(target_loc);
    writes.push_back(IORequest(target_sector * SECTOR_LEN, size_per_io_graph_, page_buf_map[target_sector], 0, 0));
    for (auto &kv : page_buf_map) {
      if (kv.first == target_sector) continue;
      writes.push_back(IORequest(kv.first * SECTOR_LEN, size_per_io_graph_, kv.second, 0, 0));
    }

    // Visibility no longer requires durability here: the target vector was staged into
    // the in-memory overlay at the top of this function (stage_decouple_vector_write),
    // and decoupled search Phase-2 consults that overlay (try_read_decouple_vector)
    // before falling back to disk. So we can publish the graph edge without a synchronous
    // vector flush; the overlay is drained to disk lazily (full-sector / shutdown flush).
    if (stats) stats->ensure_vec_us = 0;

    Timer publish_timer;
    set_id2loc(target_id, target_loc);

    // Keep the slot reserved until its new graph payload is visible. Search
    // snapshots every slot owner before reading a page, including slots other
    // than the requested node. Publishing the owner first lets a reader label
    // old recycled-slot bytes as the new node, then reuse that stale snapshot.
    // id2loc must still precede new incoming edges; loc2id follows the payload.
    std::vector<uint64_t> write_page_ref;
    graph_reader_->wbc_write(writes, ctx, &write_page_ref);
    if (!split) {
      set_loc2id(target_loc, target_id);
    }

    if (split) {
      // Publish the relocation of split victims. Their data already sits at the
      // new loc (staged above); flipping id2loc under the idx write-lock makes
      // the move atomic w.r.t. concurrent searchers, which take the idx
      // read-lock before resolving id2loc (see decouple_search).
      std::vector<uint32_t> lock_neighbors = new_nhood;
      lock_neighbors.insert(lock_neighbors.end(), split_ids.begin(), split_ids.end());
      auto locked = lock_idx(idx_lock_table, target_id, lock_neighbors);
      auto page_locked = lock_page_idx(page_idx_lock_table, target_id, lock_neighbors);
      {
        std::lock_guard<std::mutex> guard(alloc_lock);
        for (uint32_t i = 0; i < split_ids.size(); ++i) {
          set_id2loc(split_ids[i], split_new_locs[i]);
          set_loc2id(split_new_locs[i], split_ids[i]);
        }
        for (auto &l : split_orig_locs) {
          if (l != target_loc) erase_loc2id(l);  // target may reuse a victim's old slot.
        }
        set_loc2id(target_loc, target_id);
      }
      unlock_page_idx(page_idx_lock_table, page_locked);
      unlock_idx(idx_lock_table, locked);
    }
    if (stats) stats->publish_cache_us = publish_timer.elapsed();

    Timer write_timer;
    graph_reader_->write(writes, ctx);
    if (stats) {
      stats->write_topo_us = write_timer.elapsed();
      stats->update_topo_us = stats->find_graph_loc_us + stats->lock_wait_us + stats->read_topo_us +
                              stats->build_topo_us + stats->ensure_vec_us + stats->publish_cache_us +
                              stats->write_topo_us;
    }

    pipeann::unlockReqs(this->page_lock_table, extra_pages_locked);
    pipeann::unlockReqs(this->page_lock_table, nbr_pages_locked);
    graph_reader_->deref(&write_page_ref, ctx);

    for (auto *buf : all_bufs) aligned_free(buf);
    graph_reader_->deref(page_ref, ctx);

    if (split) this->record_online_split_event(split_event);

    if (stats) stats->total_us = insert_timer.elapsed();

    this->push_query_buf(read_data);
    return target_id;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
