#include "aligned_file_reader.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "nbr/nbr.h"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>
#include <array>
#include <deque>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <unordered_map>
#include "utils/timer.h"
#include "utils/tsl/robin_map.h"
#include "utils.h"
#include "utils/page_cache.h"
#include "utils/prune_neighbors.h"
#include "nbr/pq_nbr.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

namespace pipeann {

  template<typename T, typename TagT>
  typename SSDIndex<T, TagT>::TopologyCompactionStats SSDIndex<T, TagT>::compact_topology_pages(
      char *resident_ptr, uint64_t first_page, uint64_t page_count) {
    TopologyCompactionStats stats;
    Timer timer;
    const uint64_t capacity = nnodes_per_sector_graph_;
    if (capacity <= 1) return stats;
    std::vector<uint32_t> pending_empty_pages;

    const uint64_t loc_end = std::min<uint64_t>(cur_loc.load(), loc2id_.size());
    std::vector<uint32_t> occupancy(DIV_ROUND_UP(loc_end, capacity), 0);
    for (uint64_t loc = 0; loc < loc_end; ++loc) {
      if (loc2id_[loc] == kAllocatedID) {
        LOG(ERROR) << "Topology compaction requires all insert reservations to drain";
        crash();
      }
      if (loc2id_[loc] != kInvalidID) ++occupancy[loc / capacity];
    }

    // A bounded cache of pages actually read in this pass. Two-hop discovery
    // may use this or the existing page cache, but never fetches a probe page.
    using PageCopy = std::array<uint32_t, SECTOR_LEN / sizeof(uint32_t)>;
    constexpr size_t kRememberedPages = 64;
    std::unordered_map<uint64_t, PageCopy> known_pages;
    std::deque<uint64_t> known_order;
    auto remember = [&](uint64_t page, const char *buf) {
      if (known_pages.find(page) == known_pages.end()) {
        if (known_pages.size() == kRememberedPages) {
          known_pages.erase(known_order.front());
          known_order.pop_front();
        }
        known_order.push_back(page);
      }
      memcpy(known_pages[page].data(), buf, SECTOR_LEN);
    };
    auto lookup = [&](uint64_t page, char *buf) {
      auto it = known_pages.find(page);
      if (it != known_pages.end()) {
        memcpy(buf, it->second.data(), SECTOR_LEN);
        return true;
      }
      return pipeann::cache.get(page, reinterpret_cast<uint8_t *>(buf));
    };

    char *buffers = nullptr;
    alloc_aligned(reinterpret_cast<void **>(&buffers), 3 * SECTOR_LEN, SECTOR_LEN);
    std::unique_ptr<char, decltype(&pipeann::aligned_free)> buffer_guard(buffers, &pipeann::aligned_free);
    char *source_buf = buffers;
    char *target_buf = buffers + SECTOR_LEN;
    char *probe_buf = buffers + 2 * SECTOR_LEN;
    void *ctx = graph_reader_->get_ctx();
    auto resident_page = [&](uint64_t page) -> char * {
      if (!resident_ptr || page < first_page || page - first_page >= page_count) return nullptr;
      return resident_ptr + (page - first_page) * SECTOR_LEN;
    };
    auto read_page = [&](uint64_t page, char *buf) {
      if (!lookup(page, buf)) {
        if (char *resident = resident_page(page)) {
          memcpy(buf, resident, SECTOR_LEN);
          ++stats.resident_hits;
        } else {
          std::vector<IORequest> reads{IORequest(page * SECTOR_LEN, SECTOR_LEN, buf, 0, 0)};
          graph_reader_->read(reads, ctx, false);
          ++stats.page_reads;
        }
      }
      remember(page, buf);
    };
    auto live_loc = [&](uint32_t id) -> uint32_t {
      if (id >= id2loc_.size()) return kInvalidID;
      const uint32_t loc = id2loc_[id];
      return loc < loc_end && loc2id_[loc] == id ? loc : kInvalidID;
    };

    for (uint64_t source_page = 0; source_page < occupancy.size(); ++source_page) {
      const uint32_t source_count = occupancy[source_page];
      if (source_count == 0 || 2ULL * source_count >= capacity) continue;
      ++stats.sources_attempted;
      read_page(source_page, source_buf);

      std::unordered_map<uint64_t, uint64_t> links;
      tsl::robin_set<uint32_t> first_hop;
      auto count_link = [&](uint32_t id) {
        const uint32_t loc = live_loc(id);
        if (loc == kInvalidID) return;
        const uint64_t page = loc / capacity;
        if (page != source_page && occupancy[page] > 0 &&
            source_count + occupancy[page] <= capacity) ++links[page];
      };
      const uint64_t source_start = source_page * capacity;
      for (uint64_t loc = source_start; loc < std::min(source_start + capacity, loc_end); ++loc) {
        const uint32_t id = loc2id_[loc];
        if (id == kInvalidID) continue;
        auto node = graph_node_from_page(source_buf, loc, id, "compact/source");
        check_graph_nnbrs(node.nnbrs, id, loc, "compact/source");
        for (uint32_t j = 0; j < node.nnbrs; ++j) {
          count_link(node.nbrs[j]);
          first_hop.insert(node.nbrs[j]);
        }
      }
      // Prefer direct graph locality. Only fall back to cached two-hop links
      // when no directly linked page has enough free slots.
      if (links.empty()) {
        for (uint32_t id : first_hop) {
          const uint32_t loc = live_loc(id);
          if (loc == kInvalidID || !lookup(loc / capacity, probe_buf)) continue;
          ++stats.two_hop_cache_hits;
          auto node = graph_node_from_page(probe_buf, loc, id, "compact/two_hop");
          check_graph_nnbrs(node.nnbrs, id, loc, "compact/two_hop");
          for (uint32_t j = 0; j < node.nnbrs; ++j) count_link(node.nbrs[j]);
        }
      }
      if (links.empty()) continue;

      uint64_t target_page = UINT64_MAX;
      uint64_t best_links = 0;
      for (const auto &candidate : links) {
        if (target_page == UINT64_MAX || candidate.second > best_links ||
            (candidate.second == best_links &&
             (occupancy[candidate.first] > occupancy[target_page] ||
              (occupancy[candidate.first] == occupancy[target_page] && candidate.first < target_page)))) {
          target_page = candidate.first;
          best_links = candidate.second;
        }
      }
      std::vector<uint32_t> target_slots;
      const uint64_t target_start = target_page * capacity;
      for (uint64_t loc = target_start; loc < std::min(target_start + capacity, loc_end); ++loc) {
        if (loc2id_[loc] == kInvalidID) target_slots.push_back(loc);
      }
      if (target_slots.size() < source_count) continue;
      read_page(target_page, target_buf);  // Only the chosen destination is fetched.

      // Count only newly internalized directed edges between these two pages.
      // Both buffers are already present; this diagnostic issues no extra I/O.
      auto cross_edges = [&](uint64_t from_page, char *from_buf, uint64_t to_page) {
        uint64_t count = 0;
        const uint64_t start = from_page * capacity;
        for (uint64_t loc = start; loc < std::min(start + capacity, loc_end); ++loc) {
          const uint32_t id = loc2id_[loc];
          if (id == kInvalidID) continue;
          auto node = graph_node_from_page(from_buf, loc, id, "compact/internal_edges");
          for (uint32_t j = 0; j < node.nnbrs; ++j) {
            const uint32_t neighbor_loc = live_loc(node.nbrs[j]);
            if (neighbor_loc != kInvalidID && neighbor_loc / capacity == to_page) ++count;
          }
        }
        return count;
      };
      const uint64_t added_internal_edges = cross_edges(source_page, source_buf, target_page) +
                                            cross_edges(target_page, target_buf, source_page);

      std::vector<std::pair<uint32_t, uint32_t>> moves;
      for (uint64_t loc = source_start; loc < std::min(source_start + capacity, loc_end); ++loc) {
        if (loc2id_[loc] == kInvalidID) continue;
        const uint32_t destination = target_slots[moves.size()];
        memcpy(target_buf + (destination % capacity) * max_graph_node_len_,
               source_buf + (loc % capacity) * max_graph_node_len_, max_graph_node_len_);
        moves.emplace_back(loc, destination);
      }
      std::vector<IORequest> writes{IORequest(target_page * SECTOR_LEN, SECTOR_LEN, target_buf, 0, 0)};
#ifndef DGAI_MERGE_COORD_EXCLUSIVE
      begin_page_write(target_page);
#endif
      graph_reader_->write(writes, ctx);
#ifndef DGAI_MERGE_COORD_EXCLUSIVE
      end_page_write(target_page);
#endif
      ++stats.page_writes;
      // Publish mappings only after the destination write completes. No source
      // write is needed: its stale bytes are unreachable once loc2id is cleared.
      for (const auto &move : moves) {
        const uint32_t id = loc2id_[move.first];
        id2loc_[id] = move.second;
        loc2id_[move.second] = id;
        loc2id_[move.first] = kInvalidID;
      }
      occupancy[target_page] += source_count;
      occupancy[source_page] = 0;
      remember(target_page, target_buf);  // Later merges must see the new contents.
      // The remembered cache is bounded: after eviction, the resident batch
      // must still contain the latest destination rather than pre-merge bytes.
      if (char *resident = resident_page(target_page)) memcpy(resident, target_buf, SECTOR_LEN);
      stats.moved_nodes += moves.size();
      stats.exact_added_internal_edges += added_internal_edges;
      pipeann::cache.invalidate(source_page);
      pipeann::cache.invalidate(target_page);
      // Defer empty_pages until after synchronize_epoch so in-flight search still
      // holding the old source loc cannot race a recycled page.
      pending_empty_pages.push_back(static_cast<uint32_t>(source_page));
      ++stats.sources_merged;
      ++stats.pages_reclaimed;
    }
#ifndef DGAI_MERGE_COORD_EXCLUSIVE
    if (!pending_empty_pages.empty()) {
      synchronize_epoch();  // G2: drain search that still holds pre-compaction locs
      for (uint32_t page : pending_empty_pages) empty_pages.push(page);
    }
#else
    for (uint32_t page : pending_empty_pages) empty_pages.push(page);
#endif
    stats.elapsed_us = timer.elapsed();
    LOG(INFO) << "Topology compaction: sources_attempted=" << stats.sources_attempted
              << ", sources_merged=" << stats.sources_merged
              << ", pages_reclaimed=" << stats.pages_reclaimed
              << ", page_reads=" << stats.page_reads << ", page_writes=" << stats.page_writes
              << ", two_hop_cache_hits=" << stats.two_hop_cache_hits
              << ", probe_page_reads=0, elapsed_ms=" << stats.elapsed_us / 1e3;
    LOG(INFO) << "COMPACTION_DIAG resident_hits=" << stats.resident_hits
              << ", moved_nodes=" << stats.moved_nodes
              << ", exact_added_internal_edges=" << stats.exact_added_internal_edges;
    return stats;
  }

  // Decoupled merge deletes: fix topology in-place and coalesce sparse pages;
  // internal IDs and vectors are unchanged.
  // Follows the same two-phase pattern as merge_deletes:
  //   Phase 1: Populate — scan graph locs, collect non-deleted neighbors of deleted nodes,
  //             count surviving nodes.
  //   Phase 2: Repair — scan graph locs again, for each surviving node replace deleted
  //             neighbors with their non-deleted neighbors, prune, and write back in batches.
  //   Cleanup: Update loc2id, id2loc, tags, metadata, cur_id, and persist tags/metadata.
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::decouple_merge_deletes(const std::string &in_path_prefix,
                                                  const std::string &out_path_prefix,
                                                  const std::vector<TagT> &deleted_nodes,
                                                  const tsl::robin_set<TagT> &deleted_nodes_set, uint32_t nthreads,
                                                  const uint32_t &n_sampled_nbrs) {
    if (nthreads == 0) {
      nthreads = this->max_nthreads;
    }
    std::cerr << "merge nthreads: " << nthreads << std::endl;

    void *graph_ctx = graph_reader_->get_ctx();
    std::ignore = out_path_prefix;

    Timer merge_timer;
    double read_io_us = 0;
    double write_io_us = 0;
    double phase1_read_us = 0;
    double phase1_cpu_us = 0;
    double phase2_read_us = 0;
    double phase2_write_us = 0;
    double phase2_compact_wall_us = 0;
    double phase2_build_nhood_thread_us = 0;
    double phase2_prune_thread_us = 0;
    uint64_t phase2_prune_trigger_count = 0;
    double phase2_materialize_thread_us = 0;
    uint64_t read_bytes = 0;
    uint64_t phase1_read_bytes = 0;
    uint64_t write_bytes = 0;
    LOG(INFO) << "Decouple merge deletes: " << deleted_nodes.size() << " nodes to delete, "
              << "cur_id: " << cur_id.load() << ", cur_loc: " << cur_loc.load();

#ifdef DGAI_MERGE_COORD_EXCLUSIVE
    // A/B baseline: pause online search for the entire merge via exclusive merge_lock.
    LOG(INFO) << "Decouple merge uses exclusive merge_lock (DGAI_MERGE_COORD_EXCLUSIVE).";
    merge_remap_exclusive_.store(true, std::memory_order_release);
    merge_lock.lock();
    struct ExclusiveMergeGuard {
      SSDIndex<T, TagT> *idx;
      explicit ExclusiveMergeGuard(SSDIndex<T, TagT> *i) : idx(i) {}
      ~ExclusiveMergeGuard() {
        idx->merge_lock.unlock();
        idx->merge_remap_exclusive_.store(false, std::memory_order_release);
      }
    } exclusive_guard(this);
#else
    LOG(INFO) << "Decouple merge uses page seqlock + reader epoch (search∥merge).";
#endif

    LOG(INFO) << "Graph page cache size before merge: " << pipeann::cache.cache.size();

    // Wait for background I/O tasks to drain (no exclusive held yet).
    while (!bg_tasks.empty()) {
      sleep(5);
    }
    // DynamicSSDIndex's _merge_lock has drained inserts for this merge.
    // Persist the previous generation of vector buffers before any ID is reused.
    flush_all_pending_decouple_vectors();
    if (deleted_nodes.empty()) {
      LOG(INFO) << "No nodes to delete; checking sparse topology pages.";
      compact_topology_pages();
      return;
    }
    ScopedDisablePqReadLock<T> disable_pq_read_lock(this->nbr_handler);

    // Preserve the initial entry as routing-only when its user tag is deleted.
    // ID-based state remains valid after this round's tombstones are cleared,
    // and permits a later ordinary node to reuse the same external tag.
    if (frozen_entry_id_ != kInvalidID &&
        routing_only_entry_id_.load(std::memory_order_acquire) == kInvalidID &&
        id2loc(frozen_entry_id_) != kInvalidID &&
        deleted_nodes_set.find(id2tag(frozen_entry_id_)) != deleted_nodes_set.end()) {
      routing_only_entry_id_.store(frozen_entry_id_, std::memory_order_release);
      LOG(INFO) << "Frozen entry became routing-only: id=" << frozen_entry_id_
                << ", tag=" << id2tag(frozen_entry_id_);
    }

    const uint64_t kNPG = nnodes_per_sector_graph_;  // always > 0 in decoupled mode
    const uint64_t kGraphIOLen = size_per_io_graph_;

    // =========================================================================
    // Phase 1: Populate. Scan the graph file, collect non-deleted neighbors of
    //           deleted nodes, count surviving nodes, and record deleted IDs.
    // =========================================================================
    libcuckoo::cuckoohash_map<uint32_t, std::vector<uint32_t>> deleted_nhoods;
    uint64_t new_npoints = 0;
    std::vector<uint32_t> deleted_ids;  // internal IDs of deleted nodes for cleanup
    deleted_ids.reserve(deleted_nodes.size());

    // Use the same 256 MiB sequential-I/O batches as Fresh's merge path.
    constexpr uint64_t SECTORS_PER_MERGE = (256ULL * 1024 * 1024) / SECTOR_LEN;
    constexpr uint64_t SECTORS_PER_POPULATE = 128;
    const uint64_t LOC_PER_MERGE = kNPG > 0 ? SECTORS_PER_MERGE * kNPG : SECTORS_PER_MERGE;
    const uint64_t LOC_PER_POPULATE =
        kNPG > 0 ? SECTORS_PER_POPULATE * kNPG : SECTORS_PER_POPULATE;

    uint64_t sectors_per_batch =
        kNPG > 0 ? LOC_PER_MERGE / kNPG : LOC_PER_MERGE * DIV_ROUND_UP(max_graph_node_len_, SECTOR_LEN);
    uint64_t buf_size_per_batch = sectors_per_batch * SECTOR_LEN;

    char *rbuf = nullptr;
    alloc_aligned((void **) &rbuf, buf_size_per_batch, SECTOR_LEN);
    LOG(INFO) << "Graph layout: max_graph_node_len=" << max_graph_node_len_
              << " nnodes_per_sector=" << kNPG
              << " size_per_io=" << kGraphIOLen
              << " buf_size_per_batch=" << buf_size_per_batch / 1024 / 1024 << "MB";

    // uint32_t populate_nthreads = std::min(nthreads, 4u);
    uint32_t populate_nthreads = nthreads;

    // ---------- Phase 1: populate ----------
    // Survivors are counted from in-memory loc2id_/tags only. Graph IO is issued
    // only for sectors that contain deleted nodes (see trigger_deletion in DGAI_0.2).
    for (uint64_t loc_st = 0; loc_st < cur_loc.load(); loc_st += LOC_PER_POPULATE) {
      uint64_t loc_ed = std::min(cur_loc.load(), loc_st + LOC_PER_POPULATE);
      uint64_t st_sector = graph_loc_sector_no(loc_st);

      struct DeletedLoc {
        uint32_t id;
        uint64_t loc;
      };
      std::vector<DeletedLoc> deleted_locs;
      deleted_locs.reserve(deleted_nodes.size());

      {
        Timer phase1_cpu_timer;
        for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
          if (loc >= loc2id_.size())
            continue;
          uint32_t id = loc2id_[loc];
          if (id == kInvalidID || id == kAllocatedID)
            continue;

          // Routing-only entry occupies a topology slot and is repaired in
          // Phase 2, but is neither an active point nor a reclaimable deletion.
          if (is_routing_only(id)) continue;
          TagT tag = id2tag(id);
          if (deleted_nodes_set.find(tag) == deleted_nodes_set.end()) {
            ++new_npoints;
            continue;
          }
          deleted_locs.push_back({id, loc});
        }
        phase1_cpu_us += phase1_cpu_timer.elapsed();
      }

      if (deleted_locs.empty()) {
        continue;
      }

      tsl::robin_set<uint64_t> page_set;
      std::vector<IORequest> read_reqs;
      read_reqs.reserve(deleted_locs.size());
      for (const auto &entry : deleted_locs) {
        deleted_ids.push_back(entry.id);
        uint64_t pid = graph_loc_sector_no(entry.loc);
        if (page_set.find(pid) != page_set.end()) {
          continue;
        }
        page_set.insert(pid);
        char *page_buf = rbuf + (pid - st_sector) * SECTOR_LEN;
        read_reqs.push_back(IORequest(pid * SECTOR_LEN, SECTOR_LEN, page_buf, 0, 0));
      }

      Timer io_timer;
      graph_reader_->read(read_reqs, graph_ctx, false);
      const double io_us = io_timer.elapsed();
      uint64_t io_bytes = 0;
      for (const auto &req : read_reqs) {
        io_bytes += req.len;
      }
      read_io_us += io_us;
      phase1_read_us += io_us;
      read_bytes += io_bytes;
      phase1_read_bytes += io_bytes;

      {
        Timer phase1_cpu_timer;
#pragma omp parallel for num_threads(populate_nthreads) schedule(dynamic, 64)
        for (int64_t i = 0; i < (int64_t) deleted_locs.size(); ++i) {
          const auto &entry = deleted_locs[i];
          uint32_t id = entry.id;
          uint64_t loc = entry.loc;
          TagT tag = id2tag(id);

          uint64_t loc_sector = graph_loc_sector_no(loc);
          auto page_rbuf = rbuf + (loc_sector - st_sector) * SECTOR_LEN;
          DiskGraphNode<T> node = graph_node_from_page(page_rbuf, loc, id, "merge/phase1/deleted");
          std::vector<uint32_t> nhood;
          for (uint32_t j = 0; j < node.nnbrs; ++j) {
            uint32_t nbr_id = node.nbrs[j];
#ifdef DGAI_STRICT_GRAPH_CHECKS
            if (unlikely(id2loc(nbr_id) == kInvalidID)) {
              LOG(ERROR) << "Phase 1: deleted node " << id << " (tag=" << tag
                         << ") references already-deleted neighbor " << nbr_id
                         << " — previous merge did not clean up properly";
              crash();
            }
#endif
            TagT nbr_tag = id2tag(nbr_id);
            if (is_routing_only(nbr_id) || deleted_nodes_set.find(nbr_tag) == deleted_nodes_set.end()) {
              nhood.push_back(nbr_id);
            }
          }
          if (nhood.size() > n_sampled_nbrs) {
            nhood.resize(n_sampled_nbrs);
          }
          deleted_nhoods.insert(id, nhood);
        }
        phase1_cpu_us += phase1_cpu_timer.elapsed();
      }
    }

    LOG(INFO) << "Phase 1 finished: new_npoints=" << new_npoints
              << " deleted_nhoods_size=" << deleted_nhoods.size()
              << " deleted_ids=" << deleted_ids.size()
              << " elapsed=" << merge_timer.elapsed() / 1e3 << "ms"
              << " read_io_us=" << (uint64_t) phase1_read_us;
    LOG(INFO) << "Decouple merge phase1 breakdown: read_io_us=" << (uint64_t) phase1_read_us
              << ", cpu_wall_us=" << (uint64_t) phase1_cpu_us
              << ", read_mb=" << (phase1_read_bytes >> 20);

    tsl::robin_set<uint32_t> deleted_id_set;
    deleted_id_set.reserve(deleted_ids.size());
    for (uint32_t id : deleted_ids) {
      deleted_id_set.insert(id);
    }

    // =========================================================================
    // Phase 2: Repair. Scan the graph file again (same rbuf, LOC_PER_MERGE
    //           batching), fix surviving nodes that reference deleted neighbors,
    //           and write back each batch immediately (in-place).
    // =========================================================================
    uint64_t resident_first_page = 0;
    uint64_t resident_page_count = 0;
    {
      for (uint64_t loc_st = 0; loc_st < cur_loc.load(); loc_st += LOC_PER_MERGE) {
        uint64_t loc_ed = std::min(cur_loc.load(), loc_st + LOC_PER_MERGE);

        uint64_t st_sector = graph_loc_sector_no(loc_st);
        uint64_t ed_sector = graph_loc_sector_no(loc_ed > 0 ? loc_ed - 1 : 0);
        if (kNPG == 0) {
          ed_sector += DIV_ROUND_UP(max_graph_node_len_, SECTOR_LEN) - 1;
        }
        uint64_t n_sectors_to_read = ed_sector - st_sector + 1;

        std::vector<IORequest> read_reqs;
        read_reqs.push_back(IORequest(st_sector * SECTOR_LEN, n_sectors_to_read * SECTOR_LEN, rbuf, 0, 0));
        Timer io_timer;
        graph_reader_->read(read_reqs, graph_ctx, false);
        // Only this successfully read batch is resident. Earlier batches have
        // already been overwritten in rbuf when the graph spans several batches.
        resident_first_page = st_sector;
        resident_page_count = n_sectors_to_read;
        const double io_us = io_timer.elapsed();
        const uint64_t io_bytes = n_sectors_to_read * SECTOR_LEN;
        read_io_us += io_us;
        phase2_read_us += io_us;
        read_bytes += io_bytes;

        std::atomic<bool> any_modified{false};
        {
          Timer compact_timer;
#pragma omp parallel for num_threads(nthreads) schedule(dynamic, 64) reduction(+:phase2_build_nhood_thread_us, phase2_prune_thread_us, phase2_materialize_thread_us, phase2_prune_trigger_count)
          for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
            if (loc >= loc2id_.size()) continue;
            uint32_t id = loc2id_[loc];
            if (id == kInvalidID || id == kAllocatedID) continue;

            if (deleted_id_set.find(id) != deleted_id_set.end()) continue;  // deleted

            uint64_t loc_sector = graph_loc_sector_no(loc);
            auto page_rbuf = rbuf + (loc_sector - st_sector) * SECTOR_LEN;
            DiskGraphNode<T> node = graph_node_from_page(page_rbuf, loc, id, "merge/phase2/repair");

            // All deleted references should have been cleaned up by the previous
            // merge.  Only current-round deletions (in deleted_nodes_set) need fixing.
            bool needs_fix = false;
            for (uint32_t j = 0; j < node.nnbrs; ++j) {
              uint32_t nbr_id = node.nbrs[j];
#ifdef DGAI_STRICT_GRAPH_CHECKS
              if (unlikely(id2loc(nbr_id) == kInvalidID)) {
                LOG(ERROR) << "Phase 2: surviving node " << id << " (tag=" << id2tag(id)
                           << ") references already-deleted neighbor " << nbr_id
                           << " — previous merge did not clean up properly";
                crash();
              }
#endif
              if (deleted_id_set.find(nbr_id) != deleted_id_set.end()) {
                needs_fix = true;
                break;
              }
            }
            if (!needs_fix) continue;

            double t0 = omp_get_wtime();
            // Repair: replace current-round deleted neighbors with their bridge sets.
            std::unordered_set<uint32_t> nhood_set;
            nhood_set.reserve(node.nnbrs + n_sampled_nbrs);
            for (uint32_t j = 0; j < node.nnbrs; ++j) {
              uint32_t nbr_id = node.nbrs[j];
              if (deleted_id_set.find(nbr_id) != deleted_id_set.end()) {
                deleted_nhoods.find_fn(nbr_id, [&nhood_set](const std::vector<uint32_t> &nhoods) {
                  nhood_set.insert(nhoods.begin(), nhoods.end());
                });
              } else {
                nhood_set.insert(nbr_id);
              }
            }
            nhood_set.erase(id);
            std::vector<uint32_t> nhood(nhood_set.begin(), nhood_set.end());
            phase2_build_nhood_thread_us += (omp_get_wtime() - t0) * 1000000.0;

            t0 = omp_get_wtime();
            if (nhood.size() > this->params.R) {
              ++phase2_prune_trigger_count;
              std::vector<float> dists(nhood.size(), 0.0f);
              std::vector<Neighbor> pool(nhood.size());
              uint8_t *pq_buf = nullptr;
              pipeann::alloc_aligned((void **) &pq_buf,
                                     nhood.size() * AbstractNeighbor<T>::MAX_BYTES_PER_NBR, 256);
              nbr_handler->compute_dists(id, nhood.data(), nhood.size(), dists.data(), pq_buf);
              for (uint32_t k = 0; k < nhood.size(); k++) {
                pool[k] = Neighbor(nhood[k], dists[k], true);
              }
              auto nbr = this->nbr_handler;
#ifdef REORDER_COMPUTE_PQ
              pipeann::prune_neighbors_pq(pool, nhood, params, metric,
                                          [nbr, pq_buf](uint32_t a, const uint32_t *ids, uint32_t n, float *dists) {
                                            nbr->compute_dists(a, ids, n, dists, pq_buf);
                                          });
#else
              pipeann::prune_neighbors(pool, nhood, params, metric,
                                       [nbr, pq_buf](uint32_t a, uint32_t b) {
                                         float d;
                                         nbr->compute_dists(a, &b, 1, &d, pq_buf);
                                         return d;
                                       });
#endif
              pipeann::aligned_free(pq_buf);
            }
            phase2_prune_thread_us += (omp_get_wtime() - t0) * 1000000.0;

            t0 = omp_get_wtime();
            check_graph_nnbrs((uint32_t) nhood.size(), id, loc, "merge/phase2/write");
            node.nnbrs = (uint32_t) nhood.size();
            memcpy(node.nbrs, nhood.data(), nhood.size() * sizeof(uint32_t));
            phase2_materialize_thread_us += (omp_get_wtime() - t0) * 1000000.0;
            any_modified = true;
          }
          phase2_compact_wall_us += compact_timer.elapsed();
        }

        // Write back immediately after processing (in-place, same buffer).
        // Seqlock around the write so search can detect torn pages and re-read;
        // dirty-page-only coalescing is deferred (still rewrite the whole chunk).
        if (any_modified) {
          std::vector<IORequest> write_reqs;
          write_reqs.push_back(IORequest(st_sector * SECTOR_LEN, n_sectors_to_read * SECTOR_LEN, rbuf, 0, 0));
#ifndef DGAI_MERGE_COORD_EXCLUSIVE
          begin_page_writes(st_sector, n_sectors_to_read);
#endif
          Timer write_timer;
          graph_reader_->write(write_reqs, graph_ctx);
#ifndef DGAI_MERGE_COORD_EXCLUSIVE
          end_page_writes(st_sector, n_sectors_to_read);
          pipeann::cache.invalidate_range(st_sector, n_sectors_to_read);
#endif
          const double io_us = write_timer.elapsed();
          write_io_us += io_us;
          phase2_write_us += io_us;
          write_bytes += io_bytes;
        }
        LOG(INFO) << "Phase 2: processed " << loc_ed << "/" << cur_loc.load() << " locs"
                  << (any_modified.load() ? " (wrote back)" : "");
      }
      LOG(INFO) << "Phase 2 (repair + write) finished, elapsed "
                << merge_timer.elapsed() / 1e3 << "ms."
                << " read_io_us=" << (uint64_t) phase2_read_us
                << " write_io_us=" << (uint64_t) phase2_write_us;
      LOG(INFO) << "Decouple merge phase2 breakdown: read_io_us=" << (uint64_t) phase2_read_us
                << ", compact_wall_us=" << (uint64_t) phase2_compact_wall_us
                << ", write_io_us=" << (uint64_t) phase2_write_us
                << ", build_nhood_thread_us=" << (uint64_t) phase2_build_nhood_thread_us
                << ", prune_thread_us=" << (uint64_t) phase2_prune_thread_us
                << ", prune_trigger_count=" << phase2_prune_trigger_count
                << ", materialize_thread_us=" << (uint64_t) phase2_materialize_thread_us;
      LOG(INFO) << "Delete total time: " << merge_timer.elapsed() / 1e6
                << " s (phase1 + phase2).";
#ifdef DGAI_MERGE_COORD_EXCLUSIVE
      pipeann::cache.clear();
      LOG(INFO) << "Graph page cache cleared after Phase 2.";
#else
      LOG(INFO) << "Graph page cache selectively invalidated after Phase 2 writes.";
      // Heal any mismatched begin/end so search cannot spin forever on odd vers.
      force_page_versions_even();
#endif
    }

    // =========================================================================
    // Phase 3: Cleanup metadata, loc2id, id2loc, tags. Update cur_id / cur_loc.
    // =========================================================================

    uint64_t final_npoints = new_npoints;

    if(memnav_) {
      // Insert/delete producers excluded by DynamicSSDIndex's merge barrier.
      auto &mt=memnav_->timing;
      MemNavTiming::Scope total_timer(mt,MemNavTiming::MergeTotal);
      {MemNavTiming::Scope remark_timer(mt,MemNavTiming::MergeRemark);
       for(auto id:deleted_ids)memnav_->erase_tag(uint64_t(id2tag(id)));}
      {MemNavTiming::Scope flush_timer(mt,MemNavTiming::MergeFlush);memnav_->flush();}
      // MemNav repairs atomically while queries run. Its own cohort grace
      // period reclaims detached slots without closing disk-reader admission.
      memnav_->repair_and_reclaim();
    }
#ifndef DGAI_MERGE_COORD_EXCLUSIVE
    // G1: drain searches that may still hold deleted IDs / pre-repair neighbors.
    LOG(INFO) << "Decouple merge: synchronize_epoch G1 (drain pre-repair search)...";
    synchronize_epoch();
    LOG(INFO) << "Decouple merge: synchronize_epoch G1 done.";
#endif

    // Clear loc2id / tags / id2loc after drainage (no page locks needed).
    {
      std::unordered_map<uint64_t, std::vector<uint32_t>> ids_by_page;
      for (uint32_t id : deleted_ids) {
        uint32_t loc = id2loc(id);
        if (loc == kInvalidID) {
          tags.erase(id);
          set_id2loc(id, kInvalidID);
          continue;
        }
        ids_by_page[graph_loc_sector_no(loc)].push_back(id);
      }
      for (auto &kv : ids_by_page) {
        const uint64_t page = kv.first;
        for (uint32_t id : kv.second) {
          uint32_t loc = id2loc(id);
          if (loc != kInvalidID && loc < loc2id_.size()) {
            loc2id_[loc] = kInvalidID;
          }
          tags.erase(id);
          set_id2loc(id, kInvalidID);
        }
        if (kNPG > 0) {
          uint64_t st = page * kNPG;
          uint64_t ed = st + kNPG;
          bool all_empty = true;
          for (uint64_t l = st; l < ed && l < loc2id_.size(); ++l) {
            if (loc2id_[l] != kInvalidID) {
              all_empty = false;
              break;
            }
          }
          if (all_empty) empty_pages.push(page);
        }
      }
    }

    // Update metadata.
    meta_.npoints = final_npoints;

    // DGAI retains its initial entry, even when it is routing-only. Losing
    // that identity or mapping is an invariant violation, not a reason to
    // choose a different entry.
    const uint32_t entry_loc = frozen_entry_id_ == kInvalidID ? kInvalidID : id2loc(frozen_entry_id_);
    if (frozen_entry_id_ == kInvalidID || meta_.entry_point_id != frozen_entry_id_ ||
        meta_.entry_point != frozen_entry_id_ || entry_loc == kInvalidID ||
        entry_loc >= loc2id_.size() || loc2id_[entry_loc] != frozen_entry_id_) {
      LOG(ERROR) << "Frozen entry invariant violated: initial_id=" << frozen_entry_id_
                 << ", current_id=" << meta_.entry_point_id << ", loc=" << entry_loc;
      crash();
    }

    // Publish reusable IDs only after repair, old-reader drainage, and mapping
    // cleanup. Keep static navigation-index IDs reserved because its graph still
    // describes the old vectors associated with those IDs.
    {
      std::lock_guard<std::mutex> guard(alloc_lock);
      for (uint32_t id : deleted_ids) {
        if (!memnav_ && mem_index_ && mem_index_->_tag_to_location.find(id) != mem_index_->_tag_to_location.end()) continue;
        free_decouple_ids_.push_back(id);
      }
      std::sort(free_decouple_ids_.begin(), free_decouple_ids_.end(), std::greater<uint32_t>());
      recycled_vector_writes_left_.clear();
      for (uint32_t id : free_decouple_ids_) ++recycled_vector_writes_left_[vec_loc_sector_no(id)];
    }
    LOG(INFO) << "Decouple live points: active=" << meta_.npoints
              << ", routing_only=" << (routing_only_entry_id_.load(std::memory_order_acquire) != kInvalidID ? 1 : 0)
              << ", entry_id=" << meta_.entry_point_id;
    LOG(INFO) << "Decouple vector reuse: reusable_slots=" << free_decouple_ids_.size()
              << ", reused_total=" << reused_vector_slots_.load()
              << ", id_high_water=" << cur_id.load();

    // IDs are not compacted: cur_id remains the high-water mark and allocation
    // consumes free_decouple_ids_ before growing it. Coalesce topology pages
    // only after deletion repair and mapping cleanup have completed.
    // compact_topology_pages takes page wrlocks around each move/write.
    compact_topology_pages(rbuf, resident_first_page, resident_page_count);
    aligned_free(rbuf);

    // Persist metadata (decouple mode does not reload after merge, so tags
    // are kept only in memory; metadata is saved for debugging/resume).
    meta_.print();
    std::string meta_file = in_path_prefix + "_disk.index.meta";
    meta_.save_to_disk_index(meta_file);

    LOG(INFO) << "Decouple merge deletes finished, elapsed: " << merge_timer.elapsed() / 1e3
              << "ms, new npoints: " << final_npoints
              << ", cur_id: " << cur_id.load() << ", cur_loc: " << cur_loc.load()
              << ", total_read_io_us=" << (uint64_t) read_io_us
              << ", total_write_io_us=" << (uint64_t) write_io_us
              << ", read_mb=" << (read_bytes >> 20)
              << ", write_mb=" << (write_bytes >> 20);
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
