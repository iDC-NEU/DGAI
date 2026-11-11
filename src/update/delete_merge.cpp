#include "aligned_file_reader.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "nbr/nbr.h"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>

#include <cerrno>
#include <cstdlib>
#include <cstring>
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
#include "nbr/pq_nbr.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"

namespace {
  using pipeann::kInvalidID;

  inline uint32_t remap_old_id(const std::vector<uint32_t> &old_to_new, uint64_t max_id, uint64_t old_id,
                               const char *site) {
#ifdef DGAI_STRICT_GRAPH_CHECKS
    if (unlikely(old_id >= max_id)) {
      LOG(ERROR) << "merge remap: old_id out of range at " << site << ": old_id=" << old_id
                 << " max_id=" << max_id;
      crash();
    }
    const uint32_t mapped = old_to_new[old_id];
    if (unlikely(mapped == kInvalidID)) {
      LOG(ERROR) << "merge remap: missing old_to_new entry at " << site << ": old_id=" << old_id;
      crash();
    }
    return mapped;
#else
    (void) max_id;
    (void) site;
    return old_to_new[old_id];
#endif
  }

  void verify_id_compact_maps(const std::vector<uint32_t> &old_to_new,
                              const std::vector<uint32_t> &new_to_old, uint64_t max_id,
                              uint64_t compact_npoints) {
#ifdef DGAI_STRICT_GRAPH_CHECKS
    uint64_t mapped_count = 0;
    for (uint64_t old_id = 0; old_id < max_id; ++old_id) {
      const uint32_t new_id = old_to_new[old_id];
      if (new_id == kInvalidID) {
        continue;
      }
      if (unlikely(new_id >= compact_npoints)) {
        LOG(ERROR) << "merge verify: old_to_new[" << old_id << "]=" << new_id
                   << " >= compact_npoints=" << compact_npoints;
        crash();
      }
      if (unlikely(new_to_old[new_id] != static_cast<uint32_t>(old_id))) {
        LOG(ERROR) << "merge verify: inverse mismatch old_id=" << old_id << " new_id=" << new_id
                   << " new_to_old[new_id]=" << new_to_old[new_id];
        crash();
      }
      ++mapped_count;
    }
    if (unlikely(mapped_count != compact_npoints)) {
      LOG(ERROR) << "merge verify: mapped_count=" << mapped_count
                 << " != compact_npoints=" << compact_npoints;
      crash();
    }
#else
    (void) old_to_new;
    (void) new_to_old;
    (void) max_id;
    (void) compact_npoints;
#endif
  }
}  // namespace

namespace pipeann {
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::merge_deletes(const std::string &in_path_prefix, const std::string &out_path_prefix,
                                        const std::vector<TagT> &deleted_nodes,
                                        const tsl::robin_set<TagT> &deleted_nodes_set, uint32_t nthreads,
                                        const uint32_t &n_sampled_nbrs) {
    if (nthreads == 0) {
      nthreads = this->max_nthreads;
    }

    void *ctx = reader->get_ctx();

    while (!bg_tasks.empty()) {
      sleep(5);  // simple way to wait for background IO thread.
    }
    ScopedDisablePqReadLock<T> disable_pq_read_lock(this->nbr_handler);
    std::string disk_index_out = out_path_prefix + "_disk.index";
    // Note that the index is immutable currently.
    // Step 1: populate neighborhoods, allocate IDs.
    libcuckoo::cuckoohash_map<uint32_t, std::vector<uint32_t>> deleted_nhoods;  // id -> nhood
    std::atomic<uint64_t> new_npoints = 0;
    const uint64_t        max_id = cur_id.load();
    std::vector<uint32_t> old_to_new(max_id, kInvalidID);
    std::vector<uint32_t> new_to_old;
    Timer delete_timer;
    double read_io_us = 0;
    double write_io_us = 0;
    double phase1_read_us = 0;
    double phase2_read_us = 0;
    double phase2_write_us = 0;
    double metadata_write_us = 0;
    double phase1_cpu_us = 0;
    double phase2_compact_wall_us = 0;
    double phase2_build_nhood_thread_us = 0;
    double phase2_prune_thread_us = 0;
    uint64_t phase2_prune_trigger_count = 0;
    double phase2_remap_thread_us = 0;
    double phase2_materialize_thread_us = 0;
    double shuffle_pq_us = 0;
    double rebuild_tags_us = 0;
    uint64_t read_bytes = 0;
    uint64_t write_bytes = 0;

    // constexpr uint64_t LOC_PER_POPULATE = 128 * 8;  // small to avoid blocking search threads.
    // process this many locs per iteration
    const uint64_t LOC_PER_MERGE = meta_.nnodes_per_sector > 0 ? ROUND_UP(131072, meta_.nnodes_per_sector) : 131072;
    const uint64_t LOC_PER_POPULATE = LOC_PER_MERGE;
    const uint64_t VEC_IN_WBUF = 2 * LOC_PER_MERGE;  // sliding window buffer holds 2 batches

    // Calculate buffer size based on whether we have large or small nodes
    uint64_t sectors_per_batch = meta_.nnodes_per_sector > 0
                                     ? LOC_PER_MERGE / meta_.nnodes_per_sector
                                     : LOC_PER_MERGE * DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN);
    uint64_t buf_size_per_batch = sectors_per_batch * SECTOR_LEN;

    char *rbuf = nullptr, *wbuf = nullptr;
    alloc_aligned((void **) &rbuf, buf_size_per_batch, SECTOR_LEN);
    alloc_aligned((void **) &wbuf, 2 * buf_size_per_batch, SECTOR_LEN);  // sliding window buffer.
    LOG(INFO) << "Cur loc: " << cur_loc.load() << ", cur ID: " << cur_id
              << ", nnodes_per_sector: " << meta_.nnodes_per_sector
              << ", buf_size_per_batch: " << buf_size_per_batch / 1024 / 1024 << "MB";

    uint32_t populate_nthreads = nthreads;//std::min(nthreads, 4u);  // restrict the flow.

    for (uint64_t loc_st = 0; loc_st < cur_loc; loc_st += LOC_PER_POPULATE) {
      uint64_t loc_ed = std::min(cur_loc.load(), loc_st + LOC_PER_POPULATE);

      // Calculate sector range to read for [loc_st, loc_ed)
      uint64_t st_sector = loc_sector_no(loc_st);
      uint64_t ed_sector = loc_sector_no(loc_ed > 0 ? loc_ed - 1 : 0);
      if (meta_.nnodes_per_sector == 0) {
        ed_sector += DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN) - 1;
      }
      uint64_t n_sectors_to_read = ed_sector - st_sector + 1;

      std::vector<IORequest> read_reqs;
      read_reqs.push_back(IORequest(st_sector * SECTOR_LEN, n_sectors_to_read * SECTOR_LEN, rbuf, 0, 0));
      Timer io_timer;
      reader->read(read_reqs, ctx, false);
      const double io_us = io_timer.elapsed();
      const uint64_t io_bytes = n_sectors_to_read * SECTOR_LEN;
      read_io_us += io_us;
      phase1_read_us += io_us;
      read_bytes += io_bytes;

      {
        Timer phase1_cpu_timer;
#pragma omp parallel for num_threads(populate_nthreads)
        for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
          // populate nhood.
          uint64_t id = loc2id(loc);
          if (id == kInvalidID) {
            continue;
          }

          uint64_t tag = id2tag(id);
          if (deleted_nodes_set.find(tag) == deleted_nodes_set.end()) {  // 2. not deleted, alloc ID.
#ifdef DGAI_STRICT_GRAPH_CHECKS
            if (unlikely(id >= max_id)) {
              LOG(ERROR) << "merge phase1: survivor id out of range: id=" << id << " max_id=" << max_id;
              crash();
            }
#endif
            const uint64_t new_id = new_npoints.fetch_add(1);
#ifdef DGAI_STRICT_GRAPH_CHECKS
            if (unlikely(new_id >= std::numeric_limits<uint32_t>::max())) {
              LOG(ERROR) << "merge phase1: compacted id overflow: new_id=" << new_id;
              crash();
            }
            if (unlikely(old_to_new[id] != kInvalidID)) {
              LOG(ERROR) << "merge phase1: duplicate old_to_new assignment for id=" << id;
              crash();
            }
#endif
            old_to_new[id] = static_cast<uint32_t>(new_id);
            continue;
          }

          // 3. deleted, populate nhoods.
          uint64_t loc_sector = loc_sector_no(loc);
          auto page_rbuf = rbuf + (loc_sector - st_sector) * SECTOR_LEN;
          DiskNode<T> node = node_from_page(page_rbuf, loc);
          std::vector<uint32_t> nhood;
          for (uint32_t i = 0; i < node.nnbrs; ++i) {
            uint32_t nbr_tag = id2tag(node.nbrs[i]);
            if (deleted_nodes_set.find(nbr_tag) == deleted_nodes_set.end()) {
              nhood.push_back(node.nbrs[i]);  // filtered neighborhoods.
            }
          }
          // sample for less space consumption.
          if (nhood.size() > n_sampled_nbrs) {
            // std::shuffle(nhood.begin(), nhood.end(), std::default_random_engine());
            nhood.resize(n_sampled_nbrs);  // nearest.
          }
          deleted_nhoods.insert(id, nhood);
        }
        phase1_cpu_us += phase1_cpu_timer.elapsed();
      }
    }
    const uint64_t compact_npoints = new_npoints.load();
    new_to_old.assign(compact_npoints, kInvalidID);
#pragma omp parallel for num_threads(populate_nthreads)
    for (int64_t old_id = 0; old_id < (int64_t) max_id; ++old_id) {
      const uint32_t new_id = old_to_new[old_id];
      if (new_id != kInvalidID) {
        new_to_old[new_id] = static_cast<uint32_t>(old_id);
      }
    }
    verify_id_compact_maps(old_to_new, new_to_old, max_id, compact_npoints);
    LOG(INFO) << "Finished populating neighborhoods, totally elapsed: " << delete_timer.elapsed() / 1e3
              << "ms, new npoints: " << compact_npoints << " read_io_us=" << (uint64_t) phase1_read_us;
    LOG(INFO) << "Deleted nodes size: " << deleted_nodes.size() << ", deleted_nhoods size: " << deleted_nhoods.size();
    LOG(INFO) << "Merge phase1 breakdown: read_io_us=" << (uint64_t) phase1_read_us
              << ", cpu_wall_us=" << (uint64_t) phase1_cpu_us;

    // Step 2: prune neighbors, populate PQ and tags.
    int fd = open(disk_index_out.c_str(), O_DIRECT | O_LARGEFILE | O_RDWR | O_CREAT, 0755);
    uint64_t wb_id = 0;
    std::atomic<uint64_t> n_used_id = 0;
    auto write_back = [&]() {
      // write one buffer.
      uint64_t buf_id = (wb_id % VEC_IN_WBUF) / LOC_PER_MERGE;
      auto b = wbuf + buf_id * buf_size_per_batch;
      std::vector<IORequest> write_reqs;
      uint64_t id_delta = std::min(LOC_PER_MERGE, n_used_id - wb_id);

      // Calculate write length: from wb_id to wb_id + id_delta - 1
      uint64_t st_sector = loc_sector_no(wb_id);
      uint64_t ed_sector = loc_sector_no(wb_id + id_delta - 1);
      if (meta_.nnodes_per_sector == 0) {
        ed_sector += DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN) - 1;
      }
      uint64_t write_len = (ed_sector - st_sector + 1) * SECTOR_LEN;

      write_reqs.push_back(IORequest(st_sector * SECTOR_LEN, write_len, b, 0, 0));
      Timer io_timer;
      reader->write_fd(fd, write_reqs, ctx);
      const double io_us = io_timer.elapsed();
      write_io_us += io_us;
      phase2_write_us += io_us;
      write_bytes += write_len;
      wb_id += id_delta;
      LOG(INFO) << "Write back " << wb_id << "/" << n_used_id << " IDs.";
    };

    std::vector<TagT> new_tags(compact_npoints);

    for (uint64_t loc_st = 0; loc_st < cur_loc; loc_st += LOC_PER_MERGE) {
      uint64_t loc_ed = std::min(cur_loc.load(), loc_st + LOC_PER_MERGE);

      // Calculate sector range to read for [loc_st, loc_ed)
      uint64_t st_sector = loc_sector_no(loc_st);
      uint64_t ed_sector = loc_sector_no(loc_ed > 0 ? loc_ed - 1 : 0);
      if (meta_.nnodes_per_sector == 0) {
        ed_sector += DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN) - 1;
      }
      uint64_t n_sectors_to_read = ed_sector - st_sector + 1;

      std::vector<IORequest> read_reqs;
      read_reqs.push_back(IORequest(st_sector * SECTOR_LEN, n_sectors_to_read * SECTOR_LEN, rbuf, 0, 0));
      Timer io_timer;
      reader->read(read_reqs, ctx, false);  // read in fd
      const double io_us = io_timer.elapsed();
      const uint64_t io_bytes = n_sectors_to_read * SECTOR_LEN;
      read_io_us += io_us;
      phase2_read_us += io_us;
      read_bytes += io_bytes;

      {
        Timer compact_timer;
#pragma omp parallel num_threads(nthreads) reduction(+:phase2_build_nhood_thread_us, phase2_prune_thread_us, phase2_remap_thread_us, phase2_materialize_thread_us, phase2_prune_trigger_count)
        {
          std::vector<uint32_t> nhood;
          std::vector<float> dists;
          std::vector<Neighbor> pool;
          nhood.reserve(params.C);
          dists.reserve(params.C);
          pool.reserve(params.C);
          auto nbr = this->nbr_handler;

#pragma omp for schedule(dynamic, 64)
          for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
            uint64_t id = loc2id(loc);
            if (id == kInvalidID) {
              continue;
            }

            uint64_t tag = id2tag(id);
            if (deleted_nodes_set.find(tag) != deleted_nodes_set.end()) {  // deleted.
              continue;
            }

            uint64_t loc_sector = loc_sector_no(loc);
            auto page_rbuf = rbuf + (loc_sector - st_sector) * SECTOR_LEN;
            DiskNode<T> node = node_from_page(page_rbuf, loc);

            bool needs_fix = false;
            for (uint32_t i = 0; i < node.nnbrs; ++i) {
              if (deleted_nodes_set.find(id2tag(node.nbrs[i])) != deleted_nodes_set.end()) {
                needs_fix = true;
                break;
              }
            }

            nhood.clear();
            if (needs_fix) {
              double t0 = omp_get_wtime();
              std::unordered_set<uint32_t> nhood_set;
              nhood_set.reserve(node.nnbrs + n_sampled_nbrs);
              for (uint32_t i = 0; i < node.nnbrs; ++i) {
                uint32_t nbr_tag = id2tag(node.nbrs[i]);
                if (deleted_nodes_set.find(nbr_tag) != deleted_nodes_set.end()) {
                  const auto &nhoods = deleted_nhoods.find(node.nbrs[i]);
                  nhood_set.insert(nhoods.begin(), nhoods.end());
                } else {
                  nhood_set.insert(node.nbrs[i]);
                }
              }
              nhood_set.erase(static_cast<uint32_t>(id));
              nhood.assign(nhood_set.begin(), nhood_set.end());
              phase2_build_nhood_thread_us += (omp_get_wtime() - t0) * 1000000.0;

              t0 = omp_get_wtime();
              if (nhood.size() > this->params.R) {
                ++phase2_prune_trigger_count;
                dists.resize(nhood.size());
                pool.resize(nhood.size());
                uint8_t *pq_buf = nullptr;
                pipeann::alloc_aligned((void **) &pq_buf,
                                       nhood.size() * AbstractNeighbor<T>::MAX_BYTES_PER_NBR, 256);
                nbr->compute_dists(static_cast<uint32_t>(id), nhood.data(), nhood.size(), dists.data(), pq_buf);
                for (uint32_t k = 0; k < nhood.size(); k++) {
                  pool[k].id = nhood[k];
                  pool[k].distance = dists[k];
                }
                pipeann::prune_neighbors(pool, nhood, params, metric,
                                         [nbr, pq_buf](uint32_t a, uint32_t b) {
                                           float dist;
                                           nbr->compute_dists(a, &b, 1, &dist, pq_buf);
                                           return dist;
                                         });
                pipeann::aligned_free(pq_buf);
              }
              phase2_prune_thread_us += (omp_get_wtime() - t0) * 1000000.0;
            } else {
              nhood.assign(node.nbrs, node.nbrs + node.nnbrs);
            }

            double t0 = omp_get_wtime();
            for (auto &nbr_id : nhood) {
              nbr_id = remap_old_id(old_to_new, max_id, nbr_id, "merge/phase2/nbr");
#ifdef DGAI_STRICT_GRAPH_CHECKS
              if (unlikely(nbr_id >= compact_npoints)) {
                LOG(ERROR) << "merge phase2: remapped neighbor out of range: nbr=" << nbr_id
                           << " compact_npoints=" << compact_npoints;
                crash();
              }
#endif
            }
            const uint32_t new_id = remap_old_id(old_to_new, max_id, id, "merge/phase2/center");
#ifdef DGAI_STRICT_GRAPH_CHECKS
            if (unlikely(new_id >= compact_npoints)) {
              LOG(ERROR) << "merge phase2: remapped center out of range: new_id=" << new_id
                         << " compact_npoints=" << compact_npoints;
              crash();
            }
#endif
            phase2_remap_thread_us += (omp_get_wtime() - t0) * 1000000.0;

            t0 = omp_get_wtime();
            uint64_t off = new_id % VEC_IN_WBUF;
            auto page_wbuf = wbuf + (loc_sector_no(off) - loc_sector_no(0)) * SECTOR_LEN;
            DiskNode<T> w_node = node_from_page(page_wbuf, new_id);

            memcpy(w_node.coords, node.coords, meta_.data_dim * sizeof(T));
            w_node.nnbrs = nhood.size();
            *(w_node.nbrs - 1) = w_node.nnbrs;
            memcpy(w_node.nbrs, nhood.data(), w_node.nnbrs * sizeof(uint32_t));
            ++n_used_id;
            new_tags[new_id] = id2tag(id);
            phase2_materialize_thread_us += (omp_get_wtime() - t0) * 1000000.0;
          }
        }
        phase2_compact_wall_us += compact_timer.elapsed();
      }

      LOG(INFO) << "Processed " << loc_ed << "/" << cur_loc << " locs, n_used_id: " << n_used_id << ".";
      if (n_used_id - wb_id >= LOC_PER_MERGE) {
        write_back();
      }
    }

    while (wb_id < n_used_id) {
      write_back();
    }
    LOG(INFO) << "Write nhoods finished, totally elapsed " << delete_timer.elapsed() / 1e3 << "ms."
              << " read_io_us=" << (uint64_t) phase2_read_us
              << " write_io_us=" << (uint64_t) phase2_write_us;
    LOG(INFO) << "Merge phase2 breakdown: read_io_us=" << (uint64_t) phase2_read_us
              << ", compact_wall_us=" << (uint64_t) phase2_compact_wall_us
              << ", write_io_us=" << (uint64_t) phase2_write_us
              << ", build_nhood_thread_us=" << (uint64_t) phase2_build_nhood_thread_us
              << ", prune_thread_us=" << (uint64_t) phase2_prune_thread_us
              << ", prune_trigger_count=" << phase2_prune_trigger_count
              << ", remap_thread_us=" << (uint64_t) phase2_remap_thread_us
              << ", materialize_thread_us=" << (uint64_t) phase2_materialize_thread_us;
    LOG(INFO) << "Delete total time: " << delete_timer.elapsed() / 1e6
              << " s (phase1 + phase2).";

    // duplicate neighbor handler.
    Timer shuffle_timer;
    auto new_nbr_handler = this->nbr_handler->shuffle(new_to_old, compact_npoints, nthreads);
    shuffle_pq_us = shuffle_timer.elapsed();

    while (deleted_nodes_set.find(id2tag(meta_.entry_point)) != deleted_nodes_set.end()) {
      LOG(INFO) << "Medoid deleted. Choosing another start node. Medoid ID: " << meta_.entry_point
                << " tag: " << id2tag(meta_.entry_point);
      const auto &nhoods = deleted_nhoods.find(meta_.entry_point);
      meta_.entry_point = nhoods[0];
    }
    close(fd);
    // free buf
    aligned_free((void *) rbuf);
    aligned_free((void *) wbuf);

    // set metadata, PQ and tags.
    merge_lock.lock();  // unlock in reload().
    // metadata.
    this->meta_.npoints = compact_npoints;
    this->meta_.entry_point = remap_old_id(old_to_new, max_id, meta_.entry_point, "merge/entry_point");
    // PQ.
    auto tmp = this->nbr_handler;
    this->nbr_handler = new_nbr_handler;
    delete tmp;
    // tags.
    Timer tags_timer;
    tags.clear();
    // no need to clear id2loc & loc2id as they are arrays.
    // out-of-bound loc2id is initialized in reload().
#pragma omp parallel for num_threads(nthreads)
    for (size_t i = 0; i < new_tags.size(); ++i) {
      tags.insert_or_assign(i, new_tags[i]);
      set_id2loc(i, i);
      set_loc2id(i, i);
    }
    rebuild_tags_us = tags_timer.elapsed();

    Timer metadata_io_timer;
    this->write_metadata_and_pq(in_path_prefix, out_path_prefix, &new_tags);
    metadata_write_us = metadata_io_timer.elapsed();
    write_io_us += metadata_write_us;
    LOG(INFO) << "Write metadata and PQ finished, totally elapsed " << delete_timer.elapsed() / 1e3 << "ms."
              << " metadata_write_us=" << (uint64_t) metadata_write_us
              << ", total_read_io_us=" << (uint64_t) read_io_us
              << ", total_write_io_us=" << (uint64_t) write_io_us
              << ", read_mb=" << (read_bytes >> 20)
              << ", write_mb=" << (write_bytes >> 20);
    LOG(INFO) << "Merge post-processing breakdown: shuffle_pq_us=" << (uint64_t) shuffle_pq_us
              << ", rebuild_tags_and_loc_maps_us=" << (uint64_t) rebuild_tags_us
              << ", metadata_write_us=" << (uint64_t) metadata_write_us;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::write_metadata_and_pq(const std::string &in_path_prefix, const std::string &out_path_prefix,
                                                std::vector<TagT> *new_tags) {
    // Use meta_ directly since it's already been updated
    meta_.print();
    std::string disk_index_out = out_path_prefix + "_disk.index";
    meta_.save_to_disk_index(disk_index_out);

    // Step 3. Write tags and PQ.
    std::vector<TagT> tags_vec;
    if (new_tags == nullptr) {
      tags_vec.resize(meta_.npoints);
      for (uint64_t i = 0; i < meta_.npoints; ++i) {
        tags_vec[i] = id2tag(i);
      }
      new_tags = &tags_vec;
    }
    pipeann::save_bin<TagT>(out_path_prefix + "_disk.index.tags", new_tags->data(), meta_.npoints, 1, 0);
    nbr_handler->save(out_path_prefix.c_str());
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::reload(const char *index_prefix, uint32_t num_threads) {
    std::string iprefix = std::string(index_prefix);
    std::string disk_index_file = iprefix + "_disk.index";
    this->disk_index_file = disk_index_file;
    this->max_nthreads = num_threads;

    reader->close();
    reader->open(disk_index_file, true, false);

    // reload metadata.
    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    this->init_metadata(meta);

    // No need to reload PQ, as it is already reloaded in merge_deletes.
    for (uint32_t i = this->meta_.npoints; i < this->cur_loc; ++i) {
      set_loc2id(i, kInvalidID);  // reset loc2id.
    }
    while (!this->empty_pages.empty()) {
      this->empty_pages.pop();
    }
    merge_lock.unlock();
    LOG(INFO) << "Reload finished, cur_id: " << this->cur_id << ", cur_loc: " << this->cur_loc;
    return;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
