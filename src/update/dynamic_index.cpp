#include "utils/timer.h"
#include "utils/tsl/robin_set.h"
#include "utils.h"
#include "dynamic_index.h"
#include <csignal>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <vector>
#include <utility>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <omp.h>
#include <shared_mutex>
#include <string>

#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <gperftools/malloc_extension.h>

#include "ssd_index.h"

#include "linux_aligned_file_reader.h"

namespace pipeann {
  template<typename T, typename TagT>
  DynamicSSDIndex<T, TagT>::DynamicSSDIndex(IndexBuildParameters &parameters, const std::string disk_prefix_in,
                                            const std::string disk_prefix_out, Distance<T> *dist,
                                            pipeann::Metric dist_metric, int search_mode, bool use_mem_index) {
    const bool decouple_mode =
        search_mode >= DECOUPLE_SEARCH && search_mode <= STARLING_DECOUPLE_SEARCH;
    const bool page_mode = search_mode == PAGE_SEARCH || search_mode == DECOUPLE_PAGE_SEARCH;

    if (decouple_mode) {
      if (!file_exists(disk_prefix_in + "_disk.vector") || !file_exists(disk_prefix_in + "_disk.index.meta")) {
        LOG(ERROR) << "Decoupled index files missing under prefix " << disk_prefix_in
                   << " (need _disk.vector, _disk.index.meta)";
        exit(-1);
      }
      if (search_mode == STARLING_DECOUPLE_SEARCH) {
        if (!file_exists(disk_prefix_in + "_disk.graph.starling.decouple") ||
            !file_exists(disk_prefix_in + "_partition.bin.starling.decouple")) {
          LOG(ERROR) << "Starling-decouple index files missing under prefix " << disk_prefix_in
                     << " (need _disk.graph.starling.decouple, _partition.bin.starling.decouple)";
          exit(-1);
        }
      } else if (!file_exists(disk_prefix_in + "_disk.graph")) {
        LOG(ERROR) << "Decoupled graph missing under prefix " << disk_prefix_in << " (need _disk.graph)";
        exit(-1);
      }
    } else if (page_mode) {
      if (!file_exists(disk_prefix_in + "_disk.index.starling") && !file_exists(disk_prefix_in + "_disk.index")) {
        LOG(ERROR) << "Page search index missing under prefix " << disk_prefix_in
                   << " (need _disk.index.starling or _disk.index)";
        exit(-1);
      }
    } else if (!file_exists(disk_prefix_in + "_disk.index")) {
      LOG(ERROR) << "Disk index file does not exist: " << disk_prefix_in << "_disk.index";
      exit(-1);
    }
    const std::string mem_suffix =
        (search_mode >= DECOUPLE_SEARCH && search_mode <= STARLING_DECOUPLE_SEARCH) ? kMemTopoIndexSuffix
                                                                                    : "_mem.index";
    if (use_mem_index && !file_exists(disk_prefix_in + mem_suffix)) {
      LOG(ERROR) << "In-memory index file does not exist: " << disk_prefix_in + mem_suffix;
      exit(-1);
    }

    this->_dist_metric = dist_metric;
    this->journal = new pipeann::Journal<TagT>(disk_prefix_out + "_journal");

    _paras_disk = parameters;
    _num_threads = parameters.num_threads;
    _beamwidth = parameters.beam_width;

    _disk_index_prefix_in = disk_prefix_in;
    _disk_index_prefix_out = disk_prefix_out;
    _dist_comp = dist;

    reader.reset(new LinuxAlignedFileReader());
    AbstractNeighbor<T> *nbr_handler = new PQNeighbor<T>(this->_dist_metric);
    _disk_index = new pipeann::SSDIndex<T, TagT>(this->_dist_metric, reader, nbr_handler, true, &_paras_disk);

#ifndef NO_POLLUTE_ORIGINAL
    std::string disk_index_prefix_shadow = _disk_index_prefix_in + "_shadow";
    _disk_index->copy_index(_disk_index_prefix_in, disk_index_prefix_shadow, search_mode);
    LOG(INFO) << "Copy disk index file to " << disk_index_prefix_shadow << "_disk.index";
    _disk_index_prefix_in = disk_index_prefix_shadow;
#endif

    if (search_mode == BEAM_SEARCH || search_mode == PAGE_SEARCH || search_mode == PIPE_SEARCH) {
      this->search_mode = search_mode;
    } else if (search_mode == DECOUPLE_SEARCH || search_mode == DECOUPLE_BEAM_SEARCH ||
               search_mode == STARLING_DECOUPLE_SEARCH) {
      this->search_mode = search_mode;
    } else {
      LOG(ERROR) << "Invalid search mode: " << search_mode
                 << ". Must be one of BEAM_SEARCH, PAGE_SEARCH, PIPE_SEARCH, DECOUPLE_SEARCH, "
                 << "DECOUPLE_BEAM_SEARCH, or STARLING_DECOUPLE_SEARCH.";
      exit(-1);
    }
    bool use_page_search = (search_mode == PAGE_SEARCH);

    if (search_mode == DECOUPLE_SEARCH || search_mode == DECOUPLE_BEAM_SEARCH ||
        search_mode == STARLING_DECOUPLE_SEARCH) {
      const std::string partition_suffix =
          (search_mode == STARLING_DECOUPLE_SEARCH) ? "starling.decouple" : "dgai";
      int res = _disk_index->load_decouple_index(_disk_index_prefix_in.c_str(), _num_threads, partition_suffix);
      if (res != 0) {
        LOG(INFO) << "Failed to load decouple disk index in DynamicSSDIndex constructor";
        exit(-1);
      }
    } else {
      int res = _disk_index->load(_disk_index_prefix_in.c_str(), _num_threads, use_page_search);
      if (res != 0) {
        LOG(INFO) << "Failed to load disk index in DynamicSSDIndex constructor";
        exit(-1);
      }
    }

    this->_use_mem_index = use_mem_index;
    if (use_mem_index) {
      std::string mem_index_path = disk_prefix_in + mem_suffix;
      LOG(INFO) << "Use static in-memory index for acceleration, path: " << mem_index_path;
      _disk_index->load_mem_index(mem_index_path);
    }
  }

  template<typename T, typename TagT>
  DynamicSSDIndex<T, TagT>::~DynamicSSDIndex() {
    // Stop the new owned maintenance worker even though legacy SSD/journal
    // ownership cleanup is outside this change. Callers must drain producers.
    if(_disk_index)_disk_index->memnav_.reset();
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::checkpoint() {
    // TODO(gh): checkpoint the index.
    journal->checkpoint();
  }

  template<typename T, typename TagT>
  int DynamicSSDIndex<T, TagT>::insert(const T *point, const TagT &tag, InsertStats *stats) {
    std::shared_lock<std::shared_timed_mutex> lock(_merge_lock);  // prevent merge during insert
    journal->append(pipeann::TxType::kInsert, tag);
    auto *deletion_set = &deletion_sets[active_delete_set];
    if (search_mode == DECOUPLE_SEARCH || search_mode == DECOUPLE_BEAM_SEARCH ||
        search_mode == STARLING_DECOUPLE_SEARCH) {
      int id=_disk_index->decouple_insert_in_place(point, tag, deletion_set, stats, search_mode);
      if(_disk_index->memnav_) {
        MemNavTiming::Scope timer(_disk_index->memnav_->timing,MemNavTiming::Admission);
        // Serialize admission with lazy_delete, including a delete that arrived
        // while the disk insert was still in progress.
        std::shared_lock<std::shared_timed_mutex> deletion_guard(delete_lock);
        bool live=deletion_sets[0].count(tag)==0 && deletion_sets[1].count(tag)==0;
        auto *pq=static_cast<PQNeighbor<T>*>(_disk_index->nbr_handler);
        _disk_index->memnav_->insert(id,uint64_t(tag),
          live&&MemNav::sample(uint64_t(tag))?pq->encode_owned(point):std::vector<uint8_t>{},live);
      }
      return id;
    } else {
      return _disk_index->insert_in_place(point, tag, deletion_set, stats);
    }
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::search(const T *query, const uint64_t K, const uint32_t mem_L, const uint64_t search_L,
                                        const uint32_t beam_width, TagT *tags, float *distances, QueryStats *stats,
                                        bool dyn_search_l) {
    EpochGuard search_lifetime(_disk_index->reader_epoch_);
    // Do NOT take shared _merge_lock here for DECOUPLE modes.
    // OnlineSearchLoad runs through merge; holding shared would starve
    // final_merge()'s unique_lock and prevent merge from ever starting.
    // Page/IO coordination during merge is via page_lock_table (page_merge_coord_).
    // _merge_lock unique still excludes inserts for the merge window.
    std::vector<TagT> result_tags(4096);
    std::vector<float> result_distances(4096);
    auto *deletion_set = &deletion_sets[active_delete_set];
    size_t n = 0;
    if (search_mode == BEAM_SEARCH) {
      n = _disk_index->beam_search(query, search_L, mem_L, search_L, result_tags.data(), result_distances.data(),
                                   beam_width, stats, deletion_set, dyn_search_l);
    } else if (search_mode == PAGE_SEARCH) {
      n = _disk_index->page_search(query, search_L, mem_L, search_L, result_tags.data(), result_distances.data(),
                                   beam_width, stats);
    } else if (search_mode == PIPE_SEARCH) {
      n = _disk_index->pipe_search(query, search_L, mem_L, search_L, result_tags.data(), result_distances.data(),
                                   beam_width, stats);
    } else if (search_mode == DECOUPLE_SEARCH || search_mode == STARLING_DECOUPLE_SEARCH) {
      std::vector<Neighbor> _dummy_expanded;  // required by interface, not used by search path
      // k_search must be the result K (not search_L): Phase-2 windowed rerank is
      // centered on k_search. Passing search_L here made overall_performance
      // (K=10, L=100) only exact-rerank [80,100) and left Top-10 in PQ order.
      n = _disk_index->decouple_search(query, K, mem_L, search_L, _dummy_expanded,
                                       beam_width, stats,
                                       result_tags.data(), result_distances.data());
    } else if (search_mode == DECOUPLE_BEAM_SEARCH) {
      std::vector<Neighbor> _dummy_expanded;
      uint32_t helper_l = static_cast<uint32_t>(std::min<uint64_t>(search_L, UINT32_MAX));
      // Keep search_L active candidates after removing the routing-only entry.
      if (_disk_index->routing_only_entry_id_.load(std::memory_order_acquire) != kInvalidID &&
          helper_l < UINT32_MAX) ++helper_l;
      _disk_index->do_decouple_beam_search(query, mem_L, helper_l, beam_width,
                                            _dummy_expanded, nullptr, nullptr, stats,
                                            deletion_set, dyn_search_l);
      for (const auto &candidate : _dummy_expanded) {
        // Filter by internal identity, not tag: a new active node may have
        // the same tag as the retained routing-only entry.
        if (_disk_index->is_routing_only(candidate.id)) continue;
        if (n == search_L) break;
        result_tags[n] = _disk_index->id2tag(candidate.id);
        result_distances[n++] = candidate.distance;
      }
    }

    std::vector<std::pair<TagT, float>> best_vec;  // <tag, distance>
    for (size_t i = 0; i < n; i++) {
      best_vec.emplace_back(result_tags[i], result_distances[i]);
    }
    std::shared_lock<std::shared_timed_mutex> lock(delete_lock);
    size_t pos = 0;

    for (auto iter : best_vec) {
      if (deletion_sets[0].find(iter.first) == deletion_sets[0].end() &&
          deletion_sets[1].find(iter.first) == deletion_sets[1].end()) {
        tags[pos] = iter.first;
        distances[pos] = iter.second;
        pos++;
      }
      if (pos == K) {
        return;
      }
    }
    // LOG(INFO) << "Failed to find enough tags after " << i + 1 << " attempts";
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::lazy_delete(const TagT &tag) {
    const bool protect_entry = search_mode == DECOUPLE_SEARCH || search_mode == DECOUPLE_BEAM_SEARCH ||
                               search_mode == STARLING_DECOUPLE_SEARCH;
    // Match final_merge's lock order before reading the entry's ID/tag mapping.
    std::shared_lock<std::shared_timed_mutex> merge_guard(_merge_lock, std::defer_lock);
    if (protect_entry) merge_guard.lock();
    std::unique_lock<std::shared_timed_mutex> lock(delete_lock);
    journal->append(pipeann::TxType::kDelete, tag);
    if(_disk_index->memnav_)_disk_index->memnav_->erase_tag(uint64_t(tag));
    if (protect_entry && _disk_index->frozen_entry_id_ != kInvalidID &&
        _disk_index->id2loc(_disk_index->frozen_entry_id_) != kInvalidID &&
        _disk_index->id2tag(_disk_index->frozen_entry_id_) == tag) {
      // Hide it immediately, before merge, without depending on a temporary
      // deletion set. The preserved node remains available for graph routing.
      _disk_index->routing_only_entry_id_.store(_disk_index->frozen_entry_id_, std::memory_order_release);
    }

    if (deletion_sets[active_delete_set].find(tag) == deletion_sets[active_delete_set].end()) {
      deletion_sets[active_delete_set].insert(tag);
      deleted_tags[active_delete_set].push_back(tag);
    }
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::final_merge(const uint32_t &nthreads, const uint32_t &n_sampled_nbrs) {
    std::unique_lock<std::shared_timed_mutex> lock(_merge_lock);  // only one merge at a time
    // _disk_index_in -> _disk_index_out
    // Before merge, only the active deletion set contains deletes.
    {
      std::unique_lock<std::shared_timed_mutex> lock(delete_lock);
      active_delete_set = !active_delete_set;
    }
    // During merge, both deletion_sets contain deletes.
    pipeann::Timer timer;
    merge(nthreads, n_sampled_nbrs);

    // After merge, clear the inactive deletion set (as they are already merged).
    // Only concurrent search & delete; no concurrent inserts as _merge_lock is held.
    {
      std::unique_lock<std::shared_timed_mutex> lock(delete_lock);
      deletion_sets[!active_delete_set].clear();
      deleted_tags[!active_delete_set].clear();
    }

    if (search_mode == DECOUPLE_SEARCH || search_mode == DECOUPLE_BEAM_SEARCH ||
        search_mode == STARLING_DECOUPLE_SEARCH) {
      // Decoupled mode: merge writes in-place, no need to swap prefixes or reload.
      // The merge_deletes already updated metadata and cleared state.
      LOG(INFO) << "Decouple merge time : " << timer.elapsed() / 1000 << " ms";
    } else {
      std::swap(_disk_index_prefix_in, _disk_index_prefix_out);
      // reload the disk index
      _disk_index->reload(_disk_index_prefix_in.c_str(), _num_threads);
      LOG(INFO) << "Merge time : " << timer.elapsed() / 1000 << " ms";
    }
    MallocExtension::instance()->ReleaseFreeMemory();  // Return free list to OS.
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::merge(const uint32_t &nthreads, const uint32_t &n_sampled_nbrs) {
    if (search_mode == DECOUPLE_SEARCH || search_mode == DECOUPLE_BEAM_SEARCH ||
        search_mode == STARLING_DECOUPLE_SEARCH) {
      _disk_index->decouple_merge_deletes(_disk_index_prefix_in, _disk_index_prefix_out,
                                          deleted_tags[1 - active_delete_set],
                                          deletion_sets[1 - active_delete_set], nthreads, n_sampled_nbrs);
    } else {
      _disk_index->merge_deletes(_disk_index_prefix_in, _disk_index_prefix_out, deleted_tags[1 - active_delete_set],
                                 deletion_sets[1 - active_delete_set], nthreads, n_sampled_nbrs);
    }
  }

  template<typename T, typename TagT>
  void DynamicSSDIndex<T, TagT>::debug_snapshot(const std::string &path) {
    auto lt = _disk_index->tags.lock_table();
    auto &meta = _disk_index->meta_;
    uint64_t data_dim = meta.data_dim;

    // Collect all valid entries.
    std::vector<std::pair<uint32_t, TagT>> entries;
    for (auto it = lt.begin(); it != lt.end(); ++it) {
      uint32_t id = it->first;
      TagT tag = it->second;
      uint32_t loc = _disk_index->id2loc(id);
      if (loc == kInvalidID) continue;
      entries.push_back({id, tag});
    }

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
      std::cerr << "debug_snapshot: cannot open " << path << std::endl;
      return;
    }

    //=== Header: index metadata ===//
    uint32_t magic = 0xDEB2;                   // 4B
    uint32_t num = (uint32_t)entries.size();    // 4B
    uint32_t dim32 = (uint32_t)data_dim;        // 4B
    uint32_t range = meta.range;                // 4B
    uint32_t entry_point = (uint32_t)meta.entry_point;  // 4B
    uint32_t max_node_len = (uint32_t)meta.max_node_len; // 4B
    uint32_t nnodes_per_sector = meta.nnodes_per_sector; // 4B
    uint8_t  decouple_mode = _disk_index->decouple_mode_ ? 1 : 0; // 1B
    // padding to align
    uint8_t  pad[3] = {0,0,0};

    out.write((const char *)&magic, sizeof(magic));
    out.write((const char *)&num, sizeof(num));
    out.write((const char *)&dim32, sizeof(dim32));
    out.write((const char *)&range, sizeof(range));
    out.write((const char *)&entry_point, sizeof(entry_point));
    out.write((const char *)&max_node_len, sizeof(max_node_len));
    out.write((const char *)&nnodes_per_sector, sizeof(nnodes_per_sector));
    out.write((const char *)&decouple_mode, sizeof(decouple_mode));
    out.write((const char *)pad, sizeof(pad));
    // header total: 4*7 + 1 + 3 = 32B

    //=== Per-node records (sorted by id) ===//
    // Sort entries by id for deterministic output.
    std::sort(entries.begin(), entries.end(),
              [](auto &a, auto &b) { return a.first < b.first; });

    size_t max_io = std::max({(size_t)_disk_index->size_per_io,
                              (size_t)_disk_index->size_per_io_graph_,
                              (size_t)_disk_index->size_per_io_vec_});
    char *buf = nullptr;
    alloc_aligned((void **)&buf, max_io, SECTOR_LEN);

    // This dump reads vectors directly from disk (pread), bypassing the search-side
    // overlay. Flush any still-staged vectors so the on-disk vector file is complete.
    if (_disk_index->decouple_mode_) {
      _disk_index->flush_all_pending_decouple_vectors();
    }

    for (auto &e : entries) {
      uint32_t id = e.first;
      TagT tag = e.second;
      uint32_t loc = _disk_index->id2loc(id);

      // --- Read graph (neighbors) --- //
      std::vector<uint32_t> nbrs;
      if (_disk_index->decouple_mode_) {
        uint64_t pid = _disk_index->graph_loc_sector_no(loc);
        int fd = static_cast<LinuxAlignedFileReader *>(_disk_index->graph_reader_.get())->file_desc;
        if (pread(fd, buf, _disk_index->size_per_io_graph_, pid * SECTOR_LEN) == (ssize_t)_disk_index->size_per_io_graph_) {
          DiskGraphNode<T> node = _disk_index->graph_node_from_page(buf, loc);
          for (uint32_t k = 0; k < node.nnbrs; ++k) nbrs.push_back(node.nbrs[k]);
        }
      } else {
        uint64_t sid = _disk_index->loc_sector_no(loc);
        int fd = static_cast<LinuxAlignedFileReader *>(_disk_index->reader.get())->file_desc;
        if (pread(fd, buf, _disk_index->size_per_io, sid * SECTOR_LEN) == (ssize_t)_disk_index->size_per_io) {
          DiskNode<T> node = _disk_index->node_from_page(buf, loc);
          for (uint32_t k = 0; k < node.nnbrs; ++k) nbrs.push_back(node.nbrs[k]);
        }
      }

      // --- Read vector --- //
      std::vector<float> vec(data_dim, 0);
      if (_disk_index->decouple_mode_) {
        uint64_t sector_no = _disk_index->vec_loc_sector_no(id);
        uint64_t u_off = _disk_index->vec_sector_offset(id);
        int fd = static_cast<LinuxAlignedFileReader *>(_disk_index->vector_reader_.get())->file_desc;
        if (pread(fd, buf, _disk_index->size_per_io_vec_, sector_no * SECTOR_LEN) == (ssize_t)_disk_index->size_per_io_vec_) {
          T *raw = (T *)(buf + u_off);
          for (size_t d = 0; d < data_dim; ++d) vec[d] = (float)raw[d];
        }
      } else {
        uint64_t sid = _disk_index->loc_sector_no(loc);
        int fd = static_cast<LinuxAlignedFileReader *>(_disk_index->reader.get())->file_desc;
        if (pread(fd, buf, _disk_index->size_per_io, sid * SECTOR_LEN) == (ssize_t)_disk_index->size_per_io) {
          DiskNode<T> node = _disk_index->node_from_page(buf, loc);
          for (size_t d = 0; d < data_dim; ++d) vec[d] = (float)node.coords[d];
        }
      }

      // Write one node record:
      // [tag(u32)] [id(u32)] [loc(u32)] [nnbrs(u32)] [nbr_ids...] [vec_float...]
      uint32_t nnbrs = (uint32_t)nbrs.size();
      out.write((const char *)&tag, sizeof(tag));
      out.write((const char *)&id, sizeof(id));
      out.write((const char *)&loc, sizeof(loc));
      out.write((const char *)&nnbrs, sizeof(nnbrs));
      if (nnbrs > 0) out.write((const char *)nbrs.data(), nnbrs * sizeof(uint32_t));
      out.write((const char *)vec.data(), data_dim * sizeof(float));
    }

    aligned_free(buf);
    out.close();
    std::cerr << "debug_snapshot: wrote " << num << " entries to " << path
              << " (data_dim=" << dim32 << " range=" << range
              << " entry_point=" << entry_point
              << " decouple=" << (int)decouple_mode << ")" << std::endl;
  }

  template class DynamicSSDIndex<float>;
  template class DynamicSSDIndex<uint8_t>;
  template class DynamicSSDIndex<int8_t>;
}  // namespace pipeann
