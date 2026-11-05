#include "aligned_file_reader.h"
#include "linux_aligned_file_reader.h"
#include "ssd_index.h"
#include <malloc.h>
#include <iomanip>
#include <sstream>

#include <omp.h>
#include <cmath>
#include "distance.h"
#include "nbr/nbr.h"
#include "ssd_index_defs.h"
#include "utils/timer.h"
#include "utils.h"
#include "utils/topology_placement.h"

#include <filesystem>
#include <limits>
#include <unistd.h>
#include <sys/syscall.h>
#include "utils/tsl/robin_set.h"

namespace pipeann {
  void DecoupleVectorWriteCombiner::stage_write(uint64_t sector_no, uint64_t u_off, const void *point,
                                                uint64_t vec_data_len, uint64_t size_per_io_vec,
                                                uint64_t nnodes_per_sector_vec, int vec_fd,
                                                AlignedFileReader *vector_reader) {
    char *sector_buf = nullptr;
    bool need_init = false;
    uint64_t num_writes = 0;
    for (;;) {
      uint64_t victim = UINT64_MAX;
      vec_lock_table_.wrlock(sector_no);
      {
        std::lock_guard<std::mutex> lk(pending_sector_bufs_mu_);
        auto it = pending_sector_bufs_.find(sector_no);
        if (it == pending_sector_bufs_.end()) {
          if (pending_sector_bufs_.size() >= kMaxPendingSectors) {
            victim = pending_sector_bufs_.begin()->first;
          } else {
            alloc_aligned((void **) &sector_buf, size_per_io_vec, SECTOR_LEN);
            it = pending_sector_bufs_.emplace(sector_no, PendingSector{sector_buf, 0}).first;
            pending_count_.fetch_add(1, std::memory_order_release);
            need_init = true;
          }
        }
        if (victim == UINT64_MAX) {
          sector_buf = it->second.buf;
          num_writes = ++it->second.num_writes;
        }
      }
      if (victim == UINT64_MAX) break;
      // Never hold two vector-sector locks: another writer may be evicting us.
      vec_lock_table_.unlock(sector_no);
      flush_sector(victim, size_per_io_vec, vector_reader);
    }

    if (need_init) {
      // Retain live vectors on a partially free page, including a short tail.
      memset(sector_buf, 0, size_per_io_vec);
      ssize_t n;
      do {
        n = pread(vec_fd, sector_buf, size_per_io_vec, sector_no * SECTOR_LEN);
      } while (n < 0 && errno == EINTR);
      if (unlikely(n < 0)) {
        LOG(ERROR) << "pread vector write buffer failed errno=" << errno;
        crash();
      }
    }

    memcpy(sector_buf + u_off, point, vec_data_len);

    // Count staged writes rather than the last slot's offset: concurrent inserts
    // can arrive out of order, and packed pages can contain padding bytes.
    if (nnodes_per_sector_vec == 0 || num_writes >= nnodes_per_sector_vec) {
      std::vector<IORequest> flush_req;
      flush_req.push_back(IORequest(sector_no * SECTOR_LEN, size_per_io_vec, sector_buf, 0, 0));
      vector_reader->write(flush_req, vector_reader->get_ctx());
      std::lock_guard<std::mutex> lk(pending_sector_bufs_mu_);
      auto it = pending_sector_bufs_.find(sector_no);
      aligned_free(it->second.buf);
      pending_sector_bufs_.erase(it);
      pending_count_.fetch_sub(1, std::memory_order_release);
    }
    vec_lock_table_.unlock(sector_no);
  }

  bool DecoupleVectorWriteCombiner::try_read_staged_vector(uint64_t sector_no, uint64_t u_off,
                                                           uint64_t vec_data_len, void *dst) {
    // Fast path: nothing staged anywhere -> the vector is on disk, skip all locking.
    if (pending_count_.load(std::memory_order_acquire) == 0) {
      return false;
    }

    // Lock order matches stage_write/flush_sector (sector lock then map mutex) to stay
    // deadlock-free. The shared sector lock keeps the buffer alive and its contents
    // stable (frees/overwrites take the exclusive sector lock) while we copy.
    vec_lock_table_.rdlock(sector_no);
    char *sector_buf = nullptr;
    {
      std::lock_guard<std::mutex> lk(pending_sector_bufs_mu_);
      auto it = pending_sector_bufs_.find(sector_no);
      if (it != pending_sector_bufs_.end()) {
        sector_buf = it->second.buf;
      }
    }
    bool hit = false;
    if (sector_buf != nullptr) {
      memcpy(dst, sector_buf + u_off, vec_data_len);
      hit = true;
    }
    vec_lock_table_.unlock(sector_no);
    return hit;
  }

  void DecoupleVectorWriteCombiner::ensure_staged_vector_on_disk(uint64_t sector_no, uint64_t u_off,
                                                                uint64_t vec_data_len, int vec_fd) {
    vec_lock_table_.wrlock(sector_no);

    char *sector_buf = nullptr;
    {
      std::lock_guard<std::mutex> lk(pending_sector_bufs_mu_);
      auto it = pending_sector_bufs_.find(sector_no);
      if (it != pending_sector_bufs_.end()) {
        sector_buf = it->second.buf;
      }
    }

    if (sector_buf != nullptr) {
      ssize_t n = pwrite(vec_fd, sector_buf + u_off, vec_data_len, sector_no * SECTOR_LEN + u_off);
      if (unlikely(n != (ssize_t) vec_data_len)) {
        LOG(ERROR) << "pwrite staged vector failed errno=" << errno;
        crash();
      }
    }
    // If stage_write already full-flushed this sector, the buffer is gone and data is on disk.

    vec_lock_table_.unlock(sector_no);
  }

  void DecoupleVectorWriteCombiner::flush_sector(uint64_t sector_no, uint64_t size_per_io_vec,
                                                 AlignedFileReader *vector_reader) {
    vec_lock_table_.wrlock(sector_no);

    char *sector_buf = nullptr;
    {
      std::lock_guard<std::mutex> lk(pending_sector_bufs_mu_);
      auto it = pending_sector_bufs_.find(sector_no);
      if (it != pending_sector_bufs_.end()) {
        sector_buf = it->second.buf;
      }
    }

    if (sector_buf != nullptr) {
      std::vector<IORequest> flush_req;
      flush_req.push_back(IORequest(sector_no * SECTOR_LEN, size_per_io_vec, sector_buf, 0, 0));
      vector_reader->write(flush_req, vector_reader->get_ctx());
      std::lock_guard<std::mutex> lk(pending_sector_bufs_mu_);
      auto it = pending_sector_bufs_.find(sector_no);
      if (it != pending_sector_bufs_.end()) {
        aligned_free(it->second.buf);
        pending_sector_bufs_.erase(it);
        pending_count_.fetch_sub(1, std::memory_order_release);
      }
    }

    vec_lock_table_.unlock(sector_no);
  }

  void DecoupleVectorWriteCombiner::flush_all(uint64_t size_per_io_vec, AlignedFileReader *vector_reader) {
    std::vector<uint64_t> sectors;
    {
      std::lock_guard<std::mutex> lk(pending_sector_bufs_mu_);
      sectors.reserve(pending_sector_bufs_.size());
      for (auto &kv : pending_sector_bufs_) {
        sectors.push_back(kv.first);
      }
    }

    for (auto sector_no : sectors) {
      flush_sector(sector_no, size_per_io_vec, vector_reader);
    }
  }

  template<typename T, typename TagT>
  SSDIndex<T, TagT>::SSDIndex(pipeann::Metric m, std::shared_ptr<AlignedFileReader> &file_reader,
                              AbstractNeighbor<T> *nbr_handler, bool tags, IndexBuildParameters *parameters)
      : reader(file_reader), nbr_handler(nbr_handler), metric(m), enable_tags(tags) {
    this->dist_cmp.reset(pipeann::get_distance_function<T>(m));

    if (parameters != nullptr) {
      this->params = *parameters;
      params.print();
    }
    LOG(INFO) << "Use " << nbr_handler->get_name() << " as neighbor handler, metric: " << get_metric_str(m);
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::record_online_split_event(const OnlineSplitEvent &event) {
    const double total_us = event.allocator_us + event.plan_us;
    std::lock_guard<std::mutex> guard(online_split_stats_mutex_);
    auto &window = online_split_window_;
    const size_t sample_slot = window.split_count % kOnlineSplitTimingSamples;
    if (window.plan_times.size() < kOnlineSplitTimingSamples) window.plan_times.push_back(total_us);
    else window.plan_times[sample_slot] = total_us;
    ++window.split_count;
    window.max_us = std::max(window.max_us, total_us);
    const size_t histogram_size = nnodes_per_sector_graph_ + 1;
    if (window.smaller_size_hist.size() < histogram_size) {
      window.smaller_size_hist.resize(histogram_size, 0);
      window.old_size_hist.resize(histogram_size, 0);
      window.new_size_hist.resize(histogram_size, 0);
    }
    ++window.smaller_size_hist.at(std::min(event.old_page_size, event.new_page_size));
    ++window.old_size_hist.at(event.old_page_size);
    ++window.new_size_hist.at(event.new_page_size);
    window.zero_old_victim_splits += event.moved_old_nodes == 0;
    window.target_in_new_page_splits += event.target_in_new_page;
#define SUM_SPLIT_FIELD(name) window.totals.name += event.name
    SUM_SPLIT_FIELD(allocator_us);
    SUM_SPLIT_FIELD(allocator_wait_us);
    SUM_SPLIT_FIELD(plan_us);
    SUM_SPLIT_FIELD(plan_wait_us);
    SUM_SPLIT_FIELD(topology_snapshot_us);
    SUM_SPLIT_FIELD(pq_distance_us);
    SUM_SPLIT_FIELD(partition_us);
    SUM_SPLIT_FIELD(pq_pairs);
    SUM_SPLIT_FIELD(pq_batches);
    SUM_SPLIT_FIELD(local_directed_edges);
    SUM_SPLIT_FIELD(zero_degree_nodes);
    SUM_SPLIT_FIELD(no_edge_fallbacks);
    SUM_SPLIT_FIELD(initializations);
    SUM_SPLIT_FIELD(sweeps);
    SUM_SPLIT_FIELD(refinement_moves);
    SUM_SPLIT_FIELD(moved_old_nodes);
#undef SUM_SPLIT_FIELD
    ++online_split_cumulative_count_;
    online_split_cumulative_pq_pairs_ += event.pq_pairs;
    online_split_cumulative_plan_us_ += total_us;
  }

  template<typename T, typename TagT>
  std::string SSDIndex<T, TagT>::take_online_split_stats_json() {
    OnlineSplitWindow window;
    uint64_t sequence, cumulative_count, cumulative_pairs;
    double cumulative_us;
    {
      std::lock_guard<std::mutex> guard(online_split_stats_mutex_);
      window = std::move(online_split_window_);
      online_split_window_ = OnlineSplitWindow{};
      sequence = ++online_split_drain_sequence_;
      cumulative_count = online_split_cumulative_count_;
      cumulative_pairs = online_split_cumulative_pq_pairs_;
      cumulative_us = online_split_cumulative_plan_us_;
    }
    std::sort(window.plan_times.begin(), window.plan_times.end());
    const uint64_t count = window.split_count;
    const size_t sample_count = window.plan_times.size();
    auto percentile = [&](double fraction) {
      if (sample_count == 0) return 0.0;
      const size_t pos = static_cast<size_t>(std::ceil(fraction * sample_count)) - 1;
      return window.plan_times[std::min(pos, sample_count - 1)];
    };
    const auto &s = window.totals;
    const double total_us = s.allocator_us + s.plan_us;
    std::ostringstream out;
    out << std::setprecision(17)
        << "{\"policy\":\"pq_weighted\",\"window_index\":" << sequence << ",\"split_count\":" << count
        << ",\"timing_scope\":\"allocator_plus_postread_plan_excludes_common_page_io\""
        << ",\"split_total_us\":" << total_us
        << ",\"split_avg_us\":" << (count ? total_us / count : 0)
        << ",\"split_sample_count\":" << sample_count
        << ",\"split_sample_limit\":" << kOnlineSplitTimingSamples
        << ",\"split_sample_policy\":\"most_recent\""
        << ",\"split_p50_us\":" << percentile(0.50)
        << ",\"split_p95_us\":" << percentile(0.95)
        << ",\"split_p99_us\":" << percentile(0.99)
        << ",\"split_max_us\":" << window.max_us
        << ",\"allocator_us\":" << s.allocator_us
        << ",\"allocator_mutex_wait_us\":" << s.allocator_wait_us
        << ",\"pq_plan_us\":" << s.plan_us
        << ",\"plan_mutex_wait_us\":" << s.plan_wait_us
        << ",\"split_work_us\":" << std::max(0.0, total_us - s.allocator_wait_us - s.plan_wait_us)
        << ",\"topology_snapshot_us\":" << s.topology_snapshot_us
        << ",\"pq_distance_us\":" << s.pq_distance_us
        << ",\"partition_us\":" << s.partition_us
        << ",\"pq_pair_evals\":" << s.pq_pairs
        << ",\"pq_batch_calls\":" << s.pq_batches
        << ",\"local_directed_edges\":" << s.local_directed_edges
        << ",\"zero_degree_nodes\":" << s.zero_degree_nodes
        << ",\"no_edge_fallbacks\":" << s.no_edge_fallbacks
        << ",\"initializations\":" << s.initializations
        << ",\"refinement_sweeps\":" << s.sweeps
        << ",\"refinement_moves\":" << s.refinement_moves
        << ",\"moved_old_nodes\":" << s.moved_old_nodes
        << ",\"zero_old_victim_splits\":" << window.zero_old_victim_splits
        << ",\"target_in_new_page_splits\":" << window.target_in_new_page_splits
        // Split planning operates solely on supplied page buffers and PQ
        // codes. These fields describe its additional I/O, not insert search.
        << ",\"split_extra_raw_vector_reads\":0,\"split_extra_topology_reads\":0"
        << ",\"loaded_empty_topology_pages\":" << loaded_empty_topology_pages_
        << ",\"cumulative_split_count\":" << cumulative_count
        << ",\"cumulative_split_total_us\":" << cumulative_us
        << ",\"cumulative_pq_pair_evals\":" << cumulative_pairs;
    auto histogram = [&](const char *name, const std::vector<uint64_t> &values) {
      out << ",\"" << name << "\":[";
      for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << values[i];
      }
      out << ']';
    };
    histogram("smaller_group_size_hist", window.smaller_size_hist);
    histogram("old_page_size_hist", window.old_size_hist);
    histogram("new_page_size_hist", window.new_size_hist);
    out << '}';
    return out.str();
  }

  template<typename T, typename TagT>
  SSDIndex<T, TagT>::~SSDIndex() {
    LOG(INFO) << "Lock table size: " << this->idx_lock_table.size();
    LOG(INFO) << "Page cache size: " << pipeann::cache.cache.size();

    if (load_flag) {
      this->destroy_buffers();
      reader->close();
    }

    if (decouple_mode_) {
      // Flush any pending decoupled vector writes before closing.
      flush_all_pending_decouple_vectors();

      if (graph_reader_) {
        graph_reader_->close();
      }
      if (vector_reader_) {
        vector_reader_->close();
      }
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::flush_all_pending_decouple_vectors() {
    vec_write_combiner_.flush_all(size_per_io_vec_, vector_reader_.get());
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::stage_decouple_vector_write(uint32_t id, const T *point, bool reused_id) {
    if (!decouple_mode_) {
      return;
    }
    const uint64_t vec_data_len = meta_.data_dim * sizeof(T);
    const uint64_t sector_no = vec_loc_sector_no(id);
    const uint64_t u_off = vec_sector_offset(id);
    const int vec_fd = static_cast<LinuxAlignedFileReader *>(vector_reader_.get())->file_desc;
    vec_write_combiner_.stage_write(sector_no, u_off, point, vec_data_len, size_per_io_vec_, nnodes_per_sector_vec_,
                                    vec_fd, vector_reader_.get());
    if (reused_id) {
      bool page_refilled = false;
      {
        std::lock_guard<std::mutex> guard(alloc_lock);
        auto it = recycled_vector_writes_left_.find(sector_no);
        if (unlikely(it == recycled_vector_writes_left_.end() || it->second == 0)) {
          LOG(ERROR) << "Missing recycled vector slot for id=" << id;
          crash();
        }
        // Decrement after staging, not allocation: writers may arrive out of order.
        if (--it->second == 0) {
          recycled_vector_writes_left_.erase(it);
          page_refilled = true;
        }
      }
      // A partly live page is full when its reclaimed slots are all refilled;
      // it need not accumulate an entire page of new writes to reach this point.
      if (page_refilled) vec_write_combiner_.flush_sector(sector_no, size_per_io_vec_, vector_reader_.get());
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::ensure_decouple_vector_on_disk(uint32_t id) {
    if (!decouple_mode_) {
      return;
    }
    const int vec_fd = static_cast<LinuxAlignedFileReader *>(vector_reader_.get())->file_desc;
    vec_write_combiner_.ensure_staged_vector_on_disk(vec_loc_sector_no(id), vec_sector_offset(id),
                                                     meta_.data_dim * sizeof(T), vec_fd);
  }

  template<typename T, typename TagT>
  bool SSDIndex<T, TagT>::try_read_decouple_vector(uint32_t id, void *dst) {
    if (!decouple_mode_ || !vec_write_combiner_.has_pending()) {
      return false;
    }
    return vec_write_combiner_.try_read_staged_vector(vec_loc_sector_no(id), vec_sector_offset(id),
                                                      meta_.data_dim * sizeof(T), dst);
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::copy_index(const std::string &prefix_in, const std::string &prefix_out, int search_mode) {
    const bool starling_decouple = search_mode == STARLING_DECOUPLE_SEARCH;
    const bool decouple_mode =
        (search_mode >= DECOUPLE_SEARCH && search_mode <= DECOUPLE_PAGE_SEARCH) || starling_decouple;
    const bool page_mode = search_mode == PAGE_SEARCH || search_mode == DECOUPLE_PAGE_SEARCH;
    LOG(INFO) << "Copying disk index from " << prefix_in << " to " << prefix_out << " (search_mode=" << search_mode
              << ")";

    auto copy_if = [](const std::string &src, const std::string &dst) {
      if (file_exists(src)) {
        std::filesystem::copy(src, dst, std::filesystem::copy_options::overwrite_existing);
      } else {
        LOG(INFO) << "Skip missing shadow source: " << src;
      }
    };

    if (starling_decouple) {
      copy_if(prefix_in + "_disk.vector", prefix_out + "_disk.vector");
      copy_if(prefix_in + "_disk.index.meta", prefix_out + "_disk.index.meta");
      copy_if(prefix_in + "_disk.graph.starling.decouple", prefix_out + "_disk.graph.starling.decouple");
      copy_if(prefix_in + "_partition.bin.starling.decouple", prefix_out + "_partition.bin.starling.decouple");
    } else if (decouple_mode) {
      copy_if(prefix_in + "_disk.graph", prefix_out + "_disk.graph");
      copy_if(prefix_in + "_disk.vector", prefix_out + "_disk.vector");
      copy_if(prefix_in + "_disk.index.meta", prefix_out + "_disk.index.meta");
      copy_if(prefix_in + "_partition.bin.dgai", prefix_out + "_partition.bin.dgai");
    } else if (page_mode) {
      // load() peeks metadata from _disk.index, then uses .starling when nnodes_per_sector > 1.
      copy_if(prefix_in + "_disk.index", prefix_out + "_disk.index");
      copy_if(prefix_in + "_disk.index.starling", prefix_out + "_disk.index.starling");
      copy_if(prefix_in + "_partition.bin.starling", prefix_out + "_partition.bin.starling");
      if (file_exists(prefix_in + "_disk.index.tags")) {
        copy_if(prefix_in + "_disk.index.tags", prefix_out + "_disk.index.tags");
      } else {
        std::filesystem::remove(prefix_out + "_disk.index.tags");
      }
    } else {
      copy_if(prefix_in + "_disk.index", prefix_out + "_disk.index");
      if (file_exists(prefix_in + "_disk.index.tags")) {
        copy_if(prefix_in + "_disk.index.tags", prefix_out + "_disk.index.tags");
      } else {
        std::filesystem::remove(prefix_out + "_disk.index.tags");
      }
    }

    // PQ tables are shared by all modes.
    this->nbr_handler->load(prefix_in.c_str());
    this->nbr_handler->save(prefix_out.c_str());
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::init_buffers(uint64_t n_threads) {
    uint64_t n_buffers = n_threads * 2;
    LOG(INFO) << "Init buffers for " << n_threads << " threads, setup " << n_buffers << " buffers.";
    this->thread_data_queue.null_T = nullptr;
    for (uint64_t i = 0; i < n_buffers; i++) {
      QueryBuffer<T> *data = new QueryBuffer<T>();
      this->init_query_buf(*data);
      this->thread_data_bufs.push_back(data);
      this->thread_data_queue.push(data);
      this->reader->register_buf(data->sector_scratch, MAX_N_SECTOR_READS * SECTOR_LEN, 0);
    }

#ifndef READ_ONLY_TESTS
    // background thread.
    LOG(INFO) << "Setup " << kBgIOThreads << " background I/O threads for insert...";
    for (int i = 0; i < kBgIOThreads; ++i) {
      bg_io_thread_[i] = new std::thread(&SSDIndex<T, TagT>::bg_io_thread, this);
      bg_io_thread_[i]->detach();
    }
#endif
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::destroy_buffers() {
#ifndef READ_ONLY_TESTS
    for (int i = 0; i < kBgIOThreads; ++i) {
      if (bg_io_thread_[i] != nullptr) {
        auto bg_task = new BgTask{
            .thread_data = nullptr, .writes = {}, .pages_to_unlock = {}, .pages_to_deref = {}, .terminate = true};
        bg_tasks.push(bg_task);
        bg_tasks.push_notify_all();
        bg_io_thread_[i] = nullptr;
      }
    }
#endif

    while (!this->thread_data_bufs.empty()) {
      auto buf = this->thread_data_bufs.back();
      pipeann::aligned_free((void *) buf->coord_scratch);
      pipeann::aligned_free((void *) buf->sector_scratch);
      pipeann::aligned_free((void *) buf->nbr_vec_scratch);
      pipeann::aligned_free((void *) buf->nbr_ctx_scratch);
      pipeann::aligned_free((void *) buf->aligned_dist_scratch);
      pipeann::aligned_free((void *) buf->aligned_query_T);
      if (buf->vec_write_buf != nullptr) {
        pipeann::aligned_free((void *) buf->vec_write_buf);
        buf->vec_write_buf = nullptr;
      }
      this->thread_data_bufs.pop_back();
      this->thread_data_queue.pop();
      delete buf;
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_mem_index(const std::string &mem_index_path) {
    if (!this->load_flag) {
      LOG(ERROR) << "Index is not loaded. Please load the index first.";
      exit(-1);
    }
    mem_index_ = std::make_unique<pipeann::Index<T, uint32_t>>(this->metric, this->meta_.data_dim);
    mem_index_->load(mem_index_path.c_str());
  }

  template<typename T, typename TagT>
  uint32_t SSDIndex<T, TagT>::mem_search_with_pq(const T *query, QueryBuffer<T> *query_buf, uint32_t mem_L,
                                                  uint32_t *tags_out, float *dists_out) {
    if (mem_L == 0 || mem_index_ == nullptr) {
      return 0;
    }

    nbr_handler->initialize_query(query, query_buf);
    float *dist_scratch = query_buf->aligned_dist_scratch;

    auto &graph = mem_index_->_final_graph;
    auto &loc2tag = mem_index_->_location_to_tag;

    std::vector<Neighbor> best_L_nodes(mem_L + 1);
    for (uint32_t i = 0; i < mem_L + 1; i++) {
      best_L_nodes[i].distance = std::numeric_limits<float>::max();
    }

    tsl::robin_set<unsigned> inserted_into_pool;
    inserted_into_pool.reserve(mem_L * 20);

    const unsigned ep = mem_index_->_ep;
    if (loc2tag.find(ep) == loc2tag.end()) {
      return 0;
    }
    const uint32_t ep_tag = loc2tag.at(ep);
    nbr_handler->compute_dists(query_buf, &ep_tag, 1);

    uint32_t l = 0;
    inserted_into_pool.insert(ep);
    best_L_nodes[l++] = Neighbor(ep, dist_scratch[0], true);

    uint32_t k = 0;
    std::vector<uint32_t> nbr_tags;
    std::vector<unsigned> nbr_locs;
    nbr_tags.reserve(64);
    nbr_locs.reserve(64);

    while (k < l) {
      uint32_t nk = l;
      if (best_L_nodes[k].flag) {
        best_L_nodes[k].flag = false;
        const unsigned n = best_L_nodes[k].id;
        auto &cur_v = graph[n];

        nbr_tags.clear();
        nbr_locs.clear();
        for (unsigned m = 0; m < cur_v.size(); ++m) {
          const unsigned id = cur_v[m];
          if (inserted_into_pool.find(id) != inserted_into_pool.end()) {
            continue;
          }
          if (loc2tag.find(id) == loc2tag.end()) {
            continue;
          }
          inserted_into_pool.insert(id);
          nbr_locs.push_back(id);
          nbr_tags.push_back(loc2tag.at(id));
        }

        if (!nbr_tags.empty()) {
          nbr_handler->compute_dists(query_buf, nbr_tags.data(), nbr_tags.size());
          for (size_t m = 0; m < nbr_tags.size(); ++m) {
            const float dist = dist_scratch[m];
            const unsigned id = nbr_locs[m];
            if (dist >= best_L_nodes[l - 1].distance && (l == mem_L)) {
              continue;
            }
            Neighbor nn(id, dist, true);
            const unsigned r = InsertIntoPool(best_L_nodes.data(), l, nn);
            if (l < mem_L) {
              ++l;
            }
            if (r < nk) {
              nk = r;
            }
          }
        }

        if (nk <= k) {
          k = nk;
        } else {
          ++k;
        }
      } else {
        ++k;
      }
    }

    const uint32_t n_results = std::min(mem_L, l);
    for (uint32_t i = 0; i < n_results; ++i) {
      tags_out[i] = loc2tag.at(best_L_nodes[i].id);
      dists_out[i] = best_L_nodes[i].distance;
    }
    return n_results;
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load(const char *index_prefix, uint32_t num_threads, bool use_page_search) {
    std::string original_disk_index_file = std::string(index_prefix) + "_disk.index";
    std::string starling_disk_index_file = std::string(index_prefix) + "_disk.index.starling";
    std::string starling_partition_file = std::string(index_prefix) + "_partition.bin.starling";

    // Page search with nnodes_per_sector <= 1 uses one node (or one multi-sector node) per page;
    // Starling relayout + partition are only needed when multiple nodes share a sector.
    std::string disk_index_file = original_disk_index_file;
    if (use_page_search) {
      SSDIndexMetadata<T> peek_meta;
      peek_meta.load_from_disk_index(original_disk_index_file);
      if (peek_meta.nnodes_per_sector > 1) {
        if (!file_exists(starling_disk_index_file)) {
          LOG(ERROR) << "Page search requires Starling-relayout index " << starling_disk_index_file
                     << " when nnodes_per_sector=" << peek_meta.nnodes_per_sector
                     << ". Generate it with SSD_BASED_PLAN/scripts/gen_partition.sh.";
          return -1;
        }
        disk_index_file = starling_disk_index_file;
      } else {
        LOG(INFO) << "nnodes_per_sector=" << peek_meta.nnodes_per_sector
                  << ", page search uses original index with equal mapping";
      }
    }
    this->disk_index_file = disk_index_file;

    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    this->init_metadata(meta);

    if (use_page_search && meta_.nnodes_per_sector > 1 && !file_exists(starling_partition_file)) {
      LOG(ERROR) << "Page search requires Starling partition map " << starling_partition_file
                 << " when nnodes_per_sector=" << meta_.nnodes_per_sector
                 << ". Generate it with SSD_BASED_PLAN/scripts/gen_partition.sh.";
      return -1;
    }

    // load nbrs (e.g., PQ)
    nbr_handler->load(index_prefix);

    // read index metadata
    // open AlignedFileReader handle to index_file
    if (!file_exists(disk_index_file)) {
      LOG(ERROR) << "Index file " << disk_index_file << " does not exist!";
      exit(-1);
    }

    this->destroy_buffers();  // in case of re-init.
#ifndef READ_ONLY_TESTS
    reader->open(disk_index_file, true, false);
#else
    reader->open(disk_index_file, false, false);
#endif
    this->init_buffers(num_threads);
    this->max_nthreads = num_threads;

    // load page layout.
    this->use_page_search_ = use_page_search;
    const std::string page_layout_suffix =
        (use_page_search && meta_.nnodes_per_sector > 1) ? "starling" : "__identity__";
    this->load_page_layout(index_prefix, meta_.nnodes_per_sector, meta_.npoints, page_layout_suffix);

    // load tags
    if (this->enable_tags) {
      std::string tag_file = original_disk_index_file + ".tags";
      LOG(INFO) << "Loading tags from " << tag_file;
      this->load_tags(tag_file);
    }

    load_flag = true;
    LOG(INFO) << "SSDIndex loaded successfully.";
    return 0;
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::load_decouple_index(const char *index_prefix, uint32_t num_threads,
                                             const std::string &partition_suffix) {
    std::string meta_file = std::string(index_prefix) + "_disk.index.meta";
    std::string disk_index_file = std::string(index_prefix) + "_disk.index";
    std::string graph_file = std::string(index_prefix) + "_disk.graph";
    std::string vector_file = std::string(index_prefix) + "_disk.vector";

    // Prefer layout-specific graph when using a non-default partition (e.g. starling.decouple).
    if (!partition_suffix.empty() && partition_suffix != "dgai") {
      graph_file = std::string(index_prefix) + "_disk.graph." + partition_suffix;
    }

    // 1. Load metadata from _disk.index.meta (single 4K sector, decoupled metadata).
    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(meta_file);
    this->init_metadata(meta);

    // Load nbrs (e.g., PQ).
    nbr_handler->load(index_prefix);

    // 2. Set up graph file.
    this->graph_file_ = graph_file;
    this->vector_file_ = vector_file;

    if (!file_exists(graph_file)) {
      LOG(ERROR) << "Graph file " << graph_file << " does not exist!";
      exit(-1);
    }
    if (!file_exists(vector_file)) {
      LOG(ERROR) << "Vector file " << vector_file << " does not exist!";
      exit(-1);
    }

    // Compute graph-specific layout parameters.
    // Graph records: [nnbrs (4B)] [nbrs (range * 4B)] [labels (label_size B)] — no vector data.
    max_graph_node_len_ = sizeof(uint32_t) + meta_.range * sizeof(uint32_t) + meta_.label_size;
    nnodes_per_sector_graph_ = SECTOR_LEN / max_graph_node_len_;  // 0 if max_graph_node_len_ > SECTOR_LEN
    size_per_io_graph_ = SECTOR_LEN * (nnodes_per_sector_graph_ > 0
                                           ? 1
                                           : DIV_ROUND_UP(max_graph_node_len_, SECTOR_LEN));

    LOG(INFO) << "Decoupled graph layout: max_graph_node_len=" << max_graph_node_len_
              << " nnodes_per_sector=" << nnodes_per_sector_graph_
              << " size_per_io=" << size_per_io_graph_
              << " (original max_node_len=" << meta_.max_node_len
              << " nnodes_per_sector=" << meta_.nnodes_per_sector << ")";

    // Create graph reader and open graph file.
    graph_reader_.reset(new LinuxAlignedFileReader());
#ifndef READ_ONLY_TESTS
    graph_reader_->open(graph_file, true, false);  // read-write for insert/merge support
#else
    graph_reader_->open(graph_file, false, false);
#endif

    // Compute vector-specific layout (same packed-record pattern).
    uint64_t vec_data_len = meta_.data_dim * sizeof(T);
    nnodes_per_sector_vec_ = SECTOR_LEN / vec_data_len;  // 0 if vec_data_len > SECTOR_LEN
    size_per_io_vec_ = SECTOR_LEN * (nnodes_per_sector_vec_ > 0
                                         ? 1
                                         : DIV_ROUND_UP(vec_data_len, SECTOR_LEN));

    LOG(INFO) << "Decoupled vector layout: vec_data_len=" << vec_data_len
              << " nnodes_per_sector=" << nnodes_per_sector_vec_
              << " size_per_io=" << size_per_io_vec_;

    // Create vector reader and open vector file.
    vector_reader_.reset(new LinuxAlignedFileReader());
#ifndef READ_ONLY_TESTS
    vector_reader_->open(vector_file, true, false);  // read-write for insert/merge support
#else
    vector_reader_->open(vector_file, false, false);
#endif

    // 3. Initialize buffers.
    this->destroy_buffers();  // in case of re-init.
    // We still use the original reader for buffer registration (compatibility).
    // The graph_reader_ also needs buffer registration.
#ifndef READ_ONLY_TESTS
    reader->open(disk_index_file, true, false);
#else
    reader->open(disk_index_file, false, false);
#endif
    this->init_buffers(num_threads);
    this->max_nthreads = num_threads;

    // Register buffers with graph_reader_ and vector_reader_ as well.
    for (auto *buf : this->thread_data_bufs) {
      graph_reader_->register_buf(buf->sector_scratch, MAX_N_SECTOR_READS * SECTOR_LEN, 0);
      vector_reader_->register_buf(buf->sector_scratch, MAX_N_SECTOR_READS * SECTOR_LEN, 0);
    }

    // 4. Load page layout (id2loc/loc2id for graph reordering via starling).
    //    In decouple mode, use the graph file's layout (graph-only records) for C validation.
    this->use_page_search_ = false;
    this->load_page_layout(index_prefix, nnodes_per_sector_graph_, meta_.npoints, partition_suffix);

    // 5. Load tags if enabled.
    if (this->enable_tags) {
      std::string tag_file = disk_index_file + ".tags";
      LOG(INFO) << "Loading tags from " << tag_file;
      this->load_tags(tag_file);
    }

    decouple_mode_ = true;
    this->rebuild_decouple_empty_pages();
    frozen_entry_id_ = meta_.entry_point_id;
    routing_only_entry_id_.store(kInvalidID, std::memory_order_release);
    load_flag = true;
    LOG(INFO) << "Decoupled SSDIndex loaded successfully.";
    LOG(INFO) << "  Meta: " << meta_file;
    LOG(INFO) << "  Graph: " << graph_file;
    LOG(INFO) << "  Vectors: " << vector_file;
    LOG(INFO) << "  Partition suffix: " << partition_suffix;
    return 0;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::rebuild_decouple_empty_pages() {
    std::lock_guard<std::mutex> guard(alloc_lock);
    while (empty_pages.pop() != kInvalidID) {}
    loaded_empty_topology_pages_ = 0;
#ifndef NO_MAPPING
    if (decouple_mode_ && nnodes_per_sector_graph_ > 0) {
      const uint64_t capacity = nnodes_per_sector_graph_;
      const uint64_t complete_pages = std::min<uint64_t>(cur_loc.load(), loc2id_.size()) / capacity;
      for (uint64_t page = 0; page < complete_pages; ++page) {
        bool empty = true;
        for (uint64_t slot = 0; slot < capacity; ++slot) {
          if (loc2id_[page * capacity + slot] != kInvalidID) { empty = false; break; }
        }
        if (empty) {
          empty_pages.push(static_cast<uint32_t>(page));
          ++loaded_empty_topology_pages_;
        }
      }
    }
#endif
    LOG(INFO) << "Restored empty topology pages from mapping: " << loaded_empty_topology_pages_;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_page_layout(const std::string &index_prefix, const uint64_t nnodes_per_sector,
                                           const uint64_t num_points, const std::string &partition_suffix) {
    std::string partition_file;
    if (!partition_suffix.empty()) {
      partition_file = index_prefix + "_partition.bin." + partition_suffix;
    } else {
      // Fallback: try suffixes in order for backward compatibility.
      partition_file = index_prefix + "_partition.bin.dgai";
      if (!file_exists(partition_file)) {
        partition_file = index_prefix + "_partition.bin.aligned";
      }
      if (!file_exists(partition_file)) {
        partition_file = index_prefix + "_partition.bin.starling";
      }
    }
    id2loc_.resize(num_points);  // pre-allocate space first.
    loc2id_.resize(cur_loc);     // fallback size, will be adjusted if partition exists.

    if (file_exists(partition_file)) {
      LOG(INFO) << "Loading partition file " << partition_file;
      std::ifstream part(partition_file);
      uint64_t C, partition_nums, nd;
      part.read((char *) &C, sizeof(uint64_t));
      part.read((char *) &partition_nums, sizeof(uint64_t));
      part.read((char *) &nd, sizeof(uint64_t));
      if (nnodes_per_sector <= 1) {
        // nnodes_per_sector == 0: one node spans multiple sectors (e.g. 2 sectors/point).
        // nnodes_per_sector == 1: one node per sector.  Partition reordering is a no-op.
        LOG(INFO) << "nnodes_per_sector=" << nnodes_per_sector
                  << ", skip partition file and use equal partition mapping";
#ifndef NO_MAPPING
#pragma omp parallel for
        for (size_t i = 0; i < num_points; ++i) {
          id2loc_[i] = i;
          loc2id_[i] = i;
        }
        for (size_t i = num_points; i < this->cur_loc; ++i) {
          loc2id_[i] = kInvalidID;
        }
#endif
      } else if (C != nnodes_per_sector) {
        LOG(ERROR) << "partition information not correct. C = " << C << " nnodes_per_sector: " << nnodes_per_sector;
        exit(-1);
      } else {
      LOG(INFO) << "Partition meta: C: " << C << " partition_nums: " << partition_nums;

      // Adjust loc2id_ to cover all locations in the partition (partition_nums * C may exceed cur_loc).
      uint64_t partition_total_loc = partition_nums * nnodes_per_sector;
      if (partition_total_loc > loc2id_.size()) {
        loc2id_.resize(partition_total_loc);
      }

      uint64_t page_offset = loc_sector_no(0);
      auto st = std::chrono::high_resolution_clock::now();

      constexpr uint64_t n_parts_per_read = 1024 * 1024;
      std::vector<unsigned> part_buf(n_parts_per_read * (1 + nnodes_per_sector));
      for (uint64_t p = 0; p < partition_nums; p += n_parts_per_read) {
        uint64_t nxt_p = std::min(p + n_parts_per_read, partition_nums);
        part.read((char *) part_buf.data(), sizeof(unsigned) * n_parts_per_read * (1 + nnodes_per_sector));
#pragma omp parallel for schedule(dynamic)
        for (uint64_t i = p; i < nxt_p; ++i) {
          uint32_t base = (i - p) * (1 + nnodes_per_sector);
          uint32_t s = part_buf[base];  // size of this partition
          for (uint32_t j = 0; j < s; ++j) {
            uint64_t id = part_buf[base + 1 + j];
            uint64_t loc = i * nnodes_per_sector + j;
            id2loc_[id] = loc;
            loc2id_[loc] = id;
          }
          for (uint32_t j = s; j < nnodes_per_sector; ++j) {
            loc2id_[i * nnodes_per_sector + j] = kInvalidID;
          }
        }
      }
      auto et = std::chrono::high_resolution_clock::now();
      LOG(INFO) << "Page layout loaded in " << std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count()
                << " ms";

      // In decoupled mode, cur_loc must be advanced past the entire partition
      // so that find_graph_location allocates new pages from free space beyond
      // the existing layout, avoiding overlap with valid partition data.
      uint64_t total_loc = partition_nums * nnodes_per_sector;
      if (total_loc > cur_loc.load()) {
        cur_loc.store(total_loc);
        LOG(INFO) << "Decoupled mode: advanced cur_loc to " << total_loc
                  << " (partition_nums=" << partition_nums
                  << " * nnodes_per_sector=" << nnodes_per_sector << ")";
      }
      }  // nnodes_per_sector > 1 && C == nnodes_per_sector
    } else {
      // "__identity__" is a sentinel meaning equal id↔loc mapping (no partition file).
      if (!partition_suffix.empty() && partition_suffix != "__identity__") {
        LOG(ERROR) << "Required partition file " << partition_file << " does not exist!";
        exit(-1);
      }
      LOG(INFO) << partition_file << " does not exist, use equal partition mapping";
// use equal mapping for id2loc and page_layout.
#ifndef NO_MAPPING
#pragma omp parallel for
      for (size_t i = 0; i < meta_.npoints; ++i) {
        id2loc_[i] = i;
        loc2id_[i] = i;
      }
      for (size_t i = meta_.npoints; i < this->cur_loc; ++i) {
        loc2id_[i] = kInvalidID;
      }
#endif
    }
    LOG(INFO) << "Page layout loaded.";
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::load_tags(const std::string &tag_file_name, size_t offset) {
    size_t tag_num, tag_dim;
    std::vector<TagT> tag_v;
    this->tags.clear();

    if (!file_exists(tag_file_name)) {
      LOG(INFO) << "Tags file not found. Using equal mapping";
      // Equal mapping are by default eliminated in tags map.
    } else {
      LOG(INFO) << "Load tags from existing file: " << tag_file_name;
      pipeann::load_bin<TagT>(tag_file_name, tag_v, tag_num, tag_dim, offset);
      tags.reserve(tag_v.size());

#pragma omp parallel for num_threads(max_nthreads)
      for (size_t i = 0; i < tag_num; ++i) {
        tags.insert_or_assign(i, tag_v[i]);
      }
      LOG(INFO) << "Loaded " << tags.size() << " tags";
    }
  }

  template<typename T, typename TagT>
  Index<T, TagT> *SSDIndex<T, TagT>::load_to_mem(const std::string &filename) {
    std::string disk_index_file = std::string(filename) + "_disk.index";
    SSDIndexMetadata<T> meta;
    meta.load_from_disk_index(disk_index_file);
    this->init_metadata(meta);  // ensure that loc_sector_no is correct.

    Index<T, TagT> *mem_index = new Index<T, TagT>(this->metric, meta.data_dim);

    mem_index->_ep = meta.entry_point;
    mem_index->range = meta.range;
    mem_index->_nd = meta.npoints;
    mem_index->resize(meta.npoints);

    reader->open(disk_index_file, false, false);

    const uint64_t kLocsPerRead = meta_.nnodes_per_sector > 0 ? ROUND_UP(131072, meta_.nnodes_per_sector)
                                                              : 131072;  // process this many locs per iteration

    // Calculate buffer size based on whether we have large or small nodes
    uint64_t sectors_per_batch = meta_.nnodes_per_sector > 0
                                     ? kLocsPerRead / meta_.nnodes_per_sector
                                     : kLocsPerRead * DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN);
    uint64_t buf_size_per_batch = sectors_per_batch * SECTOR_LEN;

    char *buf;
    pipeann::alloc_aligned((void **) &buf, buf_size_per_batch, SECTOR_LEN);

    for (uint64_t loc_st = 0; loc_st < meta.npoints; loc_st += kLocsPerRead) {
      uint64_t loc_ed = std::min(meta.npoints, loc_st + kLocsPerRead);

      // Calculate sector range to read for [loc_st, loc_ed)
      uint64_t st_sector = loc_sector_no(loc_st);
      uint64_t ed_sector = loc_sector_no(loc_ed > 0 ? loc_ed - 1 : 0);
      uint64_t n_sectors_to_read = ed_sector - st_sector + 1;

      std::vector<IORequest> read_reqs;
      read_reqs.push_back(IORequest(st_sector * SECTOR_LEN, n_sectors_to_read * SECTOR_LEN, buf, 0, 0));
      reader->read(read_reqs, reader->get_ctx(), false);

#pragma omp parallel for
      for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
        uint64_t id = loc;
#pragma omp critical
        {
          mem_index->_location_to_tag[id] = id;
          mem_index->_tag_to_location[id] = id;
        }

        uint64_t loc_sector = loc_sector_no(loc);
        auto page_rbuf = buf + (loc_sector - st_sector) * SECTOR_LEN;
        DiskNode<T> node = node_from_page(page_rbuf, loc);

        // load data and nhood.
        memcpy(mem_index->_data.data() + id * meta.data_dim, node.coords, meta.data_dim * sizeof(T));
        std::vector<uint32_t> nhood;

        for (uint32_t i = 0; i < node.nnbrs; ++i) {
          nhood.push_back(node.nbrs[i]);
        }
        mem_index->_final_graph[id] = nhood;
      }

      LOG(INFO) << "Loaded " << loc_ed << "/" << meta.npoints << " nodes.";
    }
    return mem_index;
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::get_vector_by_id(const uint32_t &id, T *vector_coords) {
    if (!enable_tags) {
      LOG(INFO) << "Tags are disabled, cannot retrieve vector";
      return -1;
    }
    uint32_t pos = id;
    auto loc = id2loc(pos);

    size_t num_sectors = loc_sector_no(loc);
    std::ifstream disk_reader(disk_index_file.c_str(), std::ios::binary);
    std::unique_ptr<char[]> sector_buf = std::make_unique<char[]>(size_per_io);
    disk_reader.seekg(SECTOR_LEN * num_sectors, std::ios::beg);
    disk_reader.read(sector_buf.get(), size_per_io);
    DiskNode<T> node = node_from_page(sector_buf.get(), loc);
    memcpy((void *) vector_coords, (void *) node.coords, meta_.data_dim * sizeof(T));
    return 0;
  }

  template<typename T, typename TagT>
  TagT SSDIndex<T, TagT>::id2tag(uint32_t id) {
#ifdef NO_MAPPING
    return id;  // use ID to replace tags.
#else
    TagT ret;
    if (tags.find(id, ret)) {
      return ret;
    } else {
      return id;
    }
#endif
  }

  template<typename T, typename TagT>
  uint32_t SSDIndex<T, TagT>::id2loc(uint32_t id) {
#ifdef NO_MAPPING
    return id;
#else
    id2loc_resize_mu_.lock_shared();
    if (unlikely(id >= id2loc_.size())) {
      LOG(ERROR) << "id " << id << " is out of range " << id2loc_.size();
      crash();
      return kInvalidID;
    }
    uint32_t ret = id2loc_[id];
    id2loc_resize_mu_.unlock_shared();
    return ret;
#endif
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::set_id2loc(uint32_t id, uint32_t loc) {
#ifdef NO_MAPPING
    return;
#else
    if (unlikely(id >= id2loc_.size())) {
      id2loc_resize_mu_.lock();
      if (likely(id >= id2loc_.size())) {
        id2loc_.resize(1.5 * id);
        LOG(INFO) << "Resize id2loc_ to " << id2loc_.size();
      }
      id2loc_resize_mu_.unlock();
    }
    // Here, we do not grab any locks. But no matter:
    // Here, the id2loc_.size() must > id (as it only increases).
    // So, we only need to ensure that no concurrent resize happens (use-after-free).
    id2loc_resize_mu_.lock_shared();
    id2loc_[id] = loc;
    id2loc_resize_mu_.unlock_shared();
#endif
  }

  template<typename T, typename TagT>
  uint64_t SSDIndex<T, TagT>::id2page(uint32_t id) {
    uint32_t loc = id2loc(id);
    if (loc == kInvalidID) {
      return kInvalidID;
    }
    return loc_sector_no(loc);
  }

  template<typename T, typename TagT>
  uint32_t SSDIndex<T, TagT>::loc2id(uint32_t loc) {
    loc2id_resize_mu_.lock_shared();
    if (unlikely(loc > loc2id_.size())) {
      LOG(ERROR) << "loc " << loc << " is out of range " << loc2id_.size();
      crash();
      return kInvalidID;
    }
    uint32_t ret = loc2id_[loc];
    loc2id_resize_mu_.unlock_shared();
    return ret;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::set_loc2id(uint32_t loc, uint32_t id) {
    if (unlikely(loc >= loc2id_.size())) {
      loc2id_resize_mu_.lock();
      if (likely(loc >= loc2id_.size())) {
        loc2id_.resize(1.5 * loc);
        LOG(INFO) << "Resize loc2id_ to " << loc2id_.size();
      }
      loc2id_resize_mu_.unlock();
    }
    loc2id_resize_mu_.lock_shared();
    loc2id_[loc] = id;
    loc2id_resize_mu_.unlock_shared();
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::erase_loc2id(uint32_t loc) {
    loc2id_resize_mu_.lock_shared();
    loc2id_[loc] = kInvalidID;
    const uint64_t page = decouple_mode_ ? graph_loc_sector_no(loc) : loc_sector_no(loc);
    const uint64_t count = decouple_mode_ ? nnodes_per_sector_graph_ : meta_.nnodes_per_sector;
    const uint64_t st = decouple_mode_ ? page * count : sector_to_loc(page, 0);
    bool empty = true;
    for (uint64_t i = st; i < st + count && i < loc2id_.size(); ++i) {
      if (loc2id_[i] != kInvalidID) {
        empty = false;
        break;
      }
    }
    if (empty) {
      empty_pages.push(page);
    }
    loc2id_resize_mu_.unlock_shared();
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::lock_vec(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                                   const std::vector<uint32_t> &neighbors, bool rd) {
#ifndef READ_ONLY_TESTS
    std::vector<uint32_t> to_lock;
    to_lock.assign(neighbors.begin(), neighbors.end());
    if (target != kInvalidID)
      to_lock.push_back(target);
    std::sort(to_lock.begin(), to_lock.end());
    for (auto &id : to_lock) {
      rd ? lock_table.rdlock(id) : lock_table.wrlock(id);
    }
#endif
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::unlock_vec(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                                     const std::vector<uint32_t> &neighbors) {
#ifndef READ_ONLY_TESTS
    if (target != kInvalidID)
      lock_table.unlock(target);
    for (auto &id : neighbors) {
      lock_table.unlock(id);
    }
#endif
  }

  template<typename T, typename TagT>
  std::vector<uint32_t> SSDIndex<T, TagT>::get_to_lock_idx(uint32_t target, const std::vector<uint32_t> &neighbors) {
    std::vector<uint32_t> to_lock;
    to_lock.assign(neighbors.begin(), neighbors.end());
    if (target != kInvalidID) {
      to_lock.push_back(target);
    }
    // Sort and deduplicate.
    std::sort(to_lock.begin(), to_lock.end());
    auto last = std::unique(to_lock.begin(), to_lock.end());
    to_lock.erase(last, to_lock.end());
    return to_lock;
  }

  // Lock the mapping for target/page if use_page_search == false/true.
  template<typename T, typename TagT>
  std::vector<uint32_t> SSDIndex<T, TagT>::lock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                                                    const std::vector<uint32_t> &neighbors, bool rd) {
#ifndef READ_ONLY_TESTS
    std::vector<uint32_t> to_lock = get_to_lock_idx(target, neighbors);
    if (rd) {
      for (auto &id : to_lock) {
        lock_table.rdlock(id);
      }
      return to_lock;
    }

    // For pipe/decouple search, read locks may not be acquired in ascending order.
    // Deadlock may happen for concurrent search and insert.
    // We prioritize search, so use try_lock for updates.
    while (true) {
      uint32_t locked = 0;
      for (; locked < to_lock.size(); ++locked) {
        if (lock_table.trywrlock(to_lock[locked]) != 0) {
          break;
        }
      }

      if (locked == to_lock.size()) {
        return to_lock;
      }

      while (locked > 0) {
        --locked;
        lock_table.unlock(to_lock[locked]);
      }
      thread_pause();
    }
    return to_lock;
#else
    return std::vector<uint32_t>();
#endif
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::unlock_idx(pipeann::SparseLockTable<uint64_t> &lock_table,
                                     const std::vector<uint32_t> &to_lock) {
#ifndef READ_ONLY_TESTS
    for (auto &id : to_lock) {
      lock_table.unlock(id);
    }
#endif
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::unlock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, const uint32_t &to_lock) {
#ifndef READ_ONLY_TESTS
    lock_table.unlock(to_lock);
#endif
  }

  // Two-level lock, as id2page may change before and after grabbing the lock.
  template<typename T, typename TagT>
  std::vector<uint32_t> SSDIndex<T, TagT>::lock_page_idx(pipeann::SparseLockTable<uint64_t> &lock_table,
                                                         uint32_t target, const std::vector<uint32_t> &neighbors,
                                                         bool rd) {
#ifndef READ_ONLY_TESTS
    if (!use_page_search_) {
      return std::vector<uint32_t>();
    }
    std::vector<uint32_t> to_lock(neighbors.begin(), neighbors.end());
    if (target != kInvalidID) {
      to_lock.push_back(target);
    }

    for (size_t i = 0; i < to_lock.size(); ++i) {
      to_lock[i] = id2page(to_lock[i]);
    }

    // Sort and deduplicate.
    std::sort(to_lock.begin(), to_lock.end());
    auto last = std::unique(to_lock.begin(), to_lock.end());
    to_lock.erase(last, to_lock.end());

    for (auto &id : to_lock) {
      rd ? lock_table.rdlock(id) : lock_table.wrlock(id);
    }
    return to_lock;
#else
    return std::vector<uint32_t>();
#endif
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::unlock_page_idx(pipeann::SparseLockTable<uint64_t> &lock_table,
                                          const std::vector<uint32_t> &to_lock) {
#ifndef READ_ONLY_TESTS
    if (!use_page_search_) {
      return;
    }
    for (auto &id : to_lock) {
      lock_table.unlock(id);
    }
#endif
  }

  template<typename T, typename TagT>
  typename SSDIndex<T, TagT>::PageArr SSDIndex<T, TagT>::get_page_layout(uint32_t page_no) {
    loc2id_resize_mu_.lock_shared();
    PageArr ret;
    auto st = sector_to_loc(page_no, 0);
    auto ed = meta_.nnodes_per_sector == 0 ? st + 1 : st + meta_.nnodes_per_sector;
    for (uint32_t i = st; i < ed; ++i) {
      ret.push_back(loc2id_[i]);
    }
    loc2id_resize_mu_.unlock_shared();
    return ret;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::erase_and_set_loc(const std::vector<uint64_t> &old_locs,
                                            const std::vector<uint64_t> &new_locs,
                                            const std::vector<uint32_t> &new_ids) {
    std::lock_guard<std::mutex> lock(alloc_lock);
    for (uint32_t i = 0; i < new_locs.size(); ++i) {
      set_loc2id(new_locs[i], new_ids[i]);
    }
    for (auto &l : old_locs) {
      erase_loc2id(l);
    }
  }

  // Returns <loc, need_read>.
  template<typename T, typename TagT>
  std::vector<uint64_t> SSDIndex<T, TagT>::alloc_loc(int n, const std::vector<uint64_t> &hint_pages,
                                                     std::set<uint64_t> &page_need_to_read) {
    std::lock_guard<std::mutex> lock(alloc_lock);
    std::vector<uint64_t> ret;
    int cur = 0;
    // Reuse.
    uint32_t threshold = (meta_.nnodes_per_sector + INDEX_SIZE_FACTOR - 1) / INDEX_SIZE_FACTOR;

    // 1. Use empty pages.
    uint32_t empty_page = kInvalidID;
    while ((empty_page = empty_pages.pop()) != kInvalidID) {
#ifdef NO_POLLUTE_ORIGINAL
      if (empty_page < loc_sector_no(meta_.npoints)) {
        continue;
      }
#endif
      auto st = sector_to_loc(empty_page, 0);
      auto ed = meta_.nnodes_per_sector == 0 ? st + 1 : st + meta_.nnodes_per_sector;
      for (uint32_t i = st; i < ed; ++i) {
        if (unlikely(loc2id_[i] != kInvalidID)) {
          LOG(ERROR) << "Page " << empty_page << " is not empty " << i << " " << loc2id_[i];
          crash();
        }
        loc2id_[i] = kAllocatedID;
        ret.push_back(i);
        ++cur;
        if (cur == n) {
          return ret;
        }
      }
    }

    // 2. Use hint pages.
    for (auto &p : hint_pages) {
#ifdef NO_POLLUTE_ORIGINAL
      if (p < loc_sector_no(meta_.npoints)) {
        continue;
      }
#endif
      // First, see the number of holes.
      uint32_t cnt = 0;
      auto st = sector_to_loc(p, 0);
      auto ed = meta_.nnodes_per_sector == 0 ? st + 1 : st + meta_.nnodes_per_sector;
      for (uint32_t i = st; i < ed; ++i) {
        if (loc2id_[i] == kInvalidID) {
          cnt++;
        }
      }
      if (cnt < threshold) {
        continue;
      }
      // Second, allocate them.
      if (cnt < meta_.nnodes_per_sector) {
        page_need_to_read.insert(p);
      }
      for (uint32_t i = st; i < ed; ++i) {
        if (loc2id_[i] == kInvalidID) {
          loc2id_[i] = kAllocatedID;
          ret.push_back(i);
          ++cur;
          if (cur == n) {
            return ret;
          }
        }
      }
    }

    // 3. Use new pages.
    int remaining = n - cur;
    for (int i = 0; i < remaining; i++) {
      set_loc2id(cur_loc + i, kAllocatedID);  // auto resize.
      ret.push_back(cur_loc + i);
    }

    // Ensure that cur_loc is aligned.
    // The hole will eventually be recycled using either empty page queue or hint pages.
    cur_loc += remaining;
    while (meta_.nnodes_per_sector != 0 && cur_loc % meta_.nnodes_per_sector != 0) {
      set_loc2id(cur_loc++, kInvalidID);  // auto resize.
    }
    return ret;
  }

  template<typename T, typename TagT>
  bool SSDIndex<T, TagT>::alloc_graph_loc(uint32_t target_id, const std::vector<uint32_t> &new_nhood,
                                          const std::vector<Neighbor> &placement_neighbors,
                                          std::set<uint64_t> &pages_need_to_read, std::vector<uint64_t> &locs,
                                          uint64_t &split_source_page, double *alloc_lock_wait_us,
                                          uint64_t *recycled_page) {
    Timer alloc_lock_timer;
    std::lock_guard<std::mutex> lock(alloc_lock);
    if (alloc_lock_wait_us) *alloc_lock_wait_us = alloc_lock_timer.elapsed();
    locs.clear();
    split_source_page = kInvalidID;
    const uint64_t kNPG = nnodes_per_sector_graph_;
    if (recycled_page) *recycled_page = kInvalidID;
    // The caller already holds its neighbor-page locks. Reclaimed pages may
    // have lower addresses: acquire their locks without waiting, then retain
    // the successful lock through publication and writeback in the caller.
    auto reserve_page = [&]() -> uint64_t {
      if (recycled_page) {
        size_t attempts = empty_pages.size();
        while (attempts-- > 0) {
          const uint64_t page = empty_pages.pop();
          if (page == kInvalidID) break;
          const uint64_t start = page * kNPG;
          if (start + kNPG > cur_loc.load() || start + kNPG > loc2id_.size()) continue;
          bool empty = true;
          for (uint64_t i = 0; i < kNPG; ++i) {
            if (loc2id(start + i) != kInvalidID) { empty = false; break; }
          }
          if (!empty) continue;  // stale/duplicate entries are never allocated
          if (page_lock_table.trywrlock(page) != 0) {
            empty_pages.push(page);
            continue;
          }
          *recycled_page = page;
          reused_topology_pages_.fetch_add(1, std::memory_order_relaxed);
          return start;
        }
      }
      const uint64_t start = cur_loc.load();
      for (uint64_t i = 0; i < kNPG; ++i) set_loc2id(start + i, kInvalidID);
      cur_loc += kNPG;
      while (cur_loc % kNPG != 0) set_loc2id(cur_loc++, kInvalidID);
      return start;
    };
    locs.reserve(new_nhood.size() + 1);

    for (auto id : new_nhood) {
      if (unlikely(id >= cur_id.load())) {
        LOG(ERROR) << "alloc_graph_loc: invalid neighbor id=" << id << " target_id=" << target_id;
        crash();
      }
      uint32_t loc = id2loc(id);
      if (unlikely(loc == kInvalidID)) {
        LOG(ERROR) << "alloc_graph_loc: neighbor without graph loc id=" << id << " target_id=" << target_id;
        crash();
      }
      locs.emplace_back(loc);
      pages_need_to_read.insert(graph_loc_sector_no(loc));
    }

    const auto scored_pages = topology_placement_pages(placement_neighbors,
        [this](uint32_t id) { return graph_loc_sector_no(id2loc(id)); }, kInvalidID);
    std::vector<uint64_t> hint_pages;
    for (const auto &page : scored_pages) hint_pages.push_back(page.first);

    // 1. Empty slot on a hint page.
    for (auto page : hint_pages) {
      uint64_t st = page * kNPG;
      uint64_t ed = st + kNPG;
      for (uint64_t loc = st; loc < ed; ++loc) {
        if (loc >= loc2id_.size()) break;
        if (likely(loc2id(loc) == kInvalidID)) {
          set_loc2id(loc, kAllocatedID);
          locs.push_back(loc);
          pages_need_to_read.insert(page);
          return false;
        }
      }
    }

    // 2. Reserve a destination for splitting the first hint page. Grouping
    //    requires its topology, so it happens after the caller reads the page.
    //    Disabled by DGAI_DISABLE_GRAPH_SPLIT: fall through to tail allocation instead.
#ifndef DGAI_DISABLE_GRAPH_SPLIT
    if (!hint_pages.empty()) {
      uint64_t old_page = hint_pages[0];
      uint64_t old_st = old_page * kNPG;
      uint64_t old_ed = old_st + kNPG;

      size_t live_nodes = 0;
      for (uint64_t loc = old_st; loc < old_ed; ++loc) {
        if (loc < loc2id_.size()) {
          uint32_t id = loc2id(loc);
          if (id != kInvalidID && id != kAllocatedID) ++live_nodes;
        }
      }

      if (live_nodes > 1) {
        const uint64_t new_page_start = reserve_page();
        // A reservation marker also prevents a duplicate free-page queue entry
        // from allocating this destination before the actual plan is ready.
        set_loc2id(new_page_start, kAllocatedID);
        split_source_page = old_page;
        locs.push_back(new_page_start);
        pages_need_to_read.insert(old_page);
        pages_need_to_read.insert(graph_loc_sector_no(new_page_start));
        return true;
      }
    }
#endif  // !DGAI_DISABLE_GRAPH_SPLIT

    // 3. Reclaimed page, or a fresh tail page when none is available.
    const uint64_t loc = reserve_page();
    set_loc2id(loc, kAllocatedID);
    locs.push_back(loc);
    pages_need_to_read.insert(graph_loc_sector_no(loc));
    return false;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::verify_id2loc() {
    // Verify id -> loc -> id map.
    LOG(INFO) << "ID2loc size: " << id2loc_.size() << ", cur_loc: " << cur_loc.load() << ", cur_id: " << cur_id
              << ", nnodes_per_sector: " << meta_.nnodes_per_sector;
    for (uint32_t i = 0; i < cur_id; ++i) {
      auto loc = id2loc(i);
      if (loc == kInvalidID) continue;  // deleted node (decouple mode: ID space has holes)
      if (unlikely(loc >= cur_loc.load())) {
        LOG(ERROR) << "ID2loc inconsistency at ID: " << i << ", loc: " << loc << ", cur_loc: " << cur_loc.load();
        crash();
      }
      if (unlikely(loc2id(loc) != i)) {
        LOG(ERROR) << "ID2loc inconsistency at ID: " << i << ", loc: " << id2loc(i)
                   << ", loc2id: " << loc2id(id2loc(i));
        crash();
      }
    }

    LOG(INFO) << "ID2loc consistency check passed.";

    // Verify loc2id do not contain duplicate ids.
    for (uint32_t i = 0; i < cur_loc; ++i) {
      auto id = loc2id(i);
      if (id != kInvalidID && id != kAllocatedID) {
        uint32_t loc = id2loc(id);
        if (unlikely(loc != i)) {
          LOG(ERROR) << "loc2id inconsistency at loc: " << i << ", id: " << id << ", loc: " << loc;
        }
      }
    }
    LOG(INFO) << "loc2ID consistency check passed.";
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::verify_graph_neighbor_ids() {
    const uint64_t kBatchBytes = 16 * 1024 * 1024;
    const uint64_t kMaxReport = 32;
    uint64_t nodes_scanned = 0;
    uint64_t n_invalid = 0;

    auto report = [&](uint32_t src_id, uint64_t src_loc, uint32_t nbr_id, const char *reason) {
      if (n_invalid < kMaxReport) {
        LOG(ERROR) << "Graph neighbor check: src_id=" << src_id << " src_loc=" << src_loc << " nbr_id=" << nbr_id
                   << " reason=" << reason << " cur_id=" << cur_id.load();
      }
      ++n_invalid;
    };

    auto check_node = [&](uint32_t src_id, uint64_t src_loc, uint32_t nnbrs, uint32_t *nbrs) {
      if (unlikely(nnbrs > meta_.range)) {
        report(src_id, src_loc, nnbrs, "nnbrs exceeds meta_.range");
        return;
      }
      ++nodes_scanned;
      for (uint32_t j = 0; j < nnbrs; ++j) {
        const uint32_t nbr_id = nbrs[j];
        if (unlikely(nbr_id >= cur_id.load())) {
          report(src_id, src_loc, nbr_id, "neighbor id >= cur_id");
        } else if (unlikely(id2loc(nbr_id) == kInvalidID)) {
          report(src_id, src_loc, nbr_id, "neighbor id2loc is kInvalidID (deleted/orphan)");
        }
      }
    };

    if (decouple_mode_) {
      void *ctx = graph_reader_->get_ctx();
      const uint64_t kNPG = nnodes_per_sector_graph_;
      const uint64_t kIOLen = size_per_io_graph_;
      uint64_t sectors_per_batch = kBatchBytes / SECTOR_LEN;
      if (sectors_per_batch == 0) {
        sectors_per_batch = 1;
      }
      uint64_t locs_per_batch = sectors_per_batch * kNPG;
      char *buf = nullptr;
      alloc_aligned((void **) &buf, sectors_per_batch * kIOLen, SECTOR_LEN);

      for (uint64_t loc_st = 0; loc_st < cur_loc.load(); loc_st += locs_per_batch) {
        uint64_t loc_ed = std::min(cur_loc.load(), loc_st + locs_per_batch);
        uint64_t st_sector = graph_loc_sector_no(loc_st);
        uint64_t ed_sector = graph_loc_sector_no(loc_ed > 0 ? loc_ed - 1 : 0);
        uint64_t n_sectors = ed_sector - st_sector + 1;
        std::vector<IORequest> reqs;
        reqs.push_back(IORequest(st_sector * SECTOR_LEN, n_sectors * SECTOR_LEN, buf, 0, 0));
        graph_reader_->read(reqs, ctx, false);

        for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
          if (loc >= loc2id_.size()) {
            break;
          }
          uint32_t id = loc2id_[loc];
          if (id == kInvalidID || id == kAllocatedID) {
            continue;
          }
          uint64_t loc_sector = graph_loc_sector_no(loc);
          auto page_buf = buf + (loc_sector - st_sector) * SECTOR_LEN;
          DiskGraphNode<T> node(page_buf, (uint32_t) loc, max_graph_node_len_, nnodes_per_sector_graph_, meta_.range);
          check_node(id, loc, node.nnbrs, node.nbrs);
        }
      }
      aligned_free(buf);
    } else {
      void *ctx = reader->get_ctx();
      uint64_t sectors_per_batch = kBatchBytes / SECTOR_LEN;
      if (sectors_per_batch == 0) {
        sectors_per_batch = 1;
      }
      uint64_t locs_per_batch = sectors_per_batch * meta_.nnodes_per_sector;
      if (meta_.nnodes_per_sector == 0) {
        locs_per_batch = 131072;
      }
      char *buf = nullptr;
      alloc_aligned((void **) &buf, sectors_per_batch * SECTOR_LEN, SECTOR_LEN);

      for (uint64_t loc_st = 0; loc_st < cur_loc.load(); loc_st += locs_per_batch) {
        uint64_t loc_ed = std::min(cur_loc.load(), loc_st + locs_per_batch);
        uint64_t st_sector = loc_sector_no(loc_st);
        uint64_t ed_sector = loc_sector_no(loc_ed > 0 ? loc_ed - 1 : 0);
        uint64_t n_sectors = ed_sector - st_sector + 1;
        std::vector<IORequest> reqs;
        reqs.push_back(IORequest(st_sector * SECTOR_LEN, n_sectors * SECTOR_LEN, buf, 0, 0));
        reader->read(reqs, ctx, false);

        for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
          if (loc >= loc2id_.size()) {
            break;
          }
          uint32_t id = loc2id_[loc];
          if (id == kInvalidID || id == kAllocatedID) {
            continue;
          }
          uint64_t loc_sector = loc_sector_no(loc);
          auto page_buf = buf + (loc_sector - st_sector) * SECTOR_LEN;
          DiskNode<T> node = node_from_page(page_buf, loc);
          check_node(id, loc, node.nnbrs, node.nbrs);
        }
      }
      aligned_free(buf);
    }

    if (n_invalid > 0) {
      LOG(ERROR) << "Graph neighbor-id check failed: " << n_invalid << " invalid edge(s), " << nodes_scanned
                 << " live node(s) scanned (showing up to " << kMaxReport << " samples)";
      crash();
    }
    LOG(INFO) << "Graph neighbor-id check passed (" << nodes_scanned << " live nodes scanned).";
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::log_graph_stats() {
    uint64_t total_degree = 0, total_nodes = 0, min_deg = params.R, max_deg = 0;
    uint64_t low_deg = 0, isolated = 0;  // low: < R/2, isolated: == 0
    const uint64_t kBatchBytes = 16 * 1024 * 1024;  // 16MB batch

    if (decouple_mode_) {
      void *ctx = graph_reader_->get_ctx();
      const uint64_t kNPG = nnodes_per_sector_graph_;
      const uint64_t kIOLen = size_per_io_graph_;
      uint64_t sectors_per_batch = kBatchBytes / SECTOR_LEN;
      if (sectors_per_batch == 0) sectors_per_batch = 1;
      uint64_t locs_per_batch = sectors_per_batch * kNPG;
      char *buf = nullptr;
      alloc_aligned((void **) &buf, sectors_per_batch * kIOLen, SECTOR_LEN);

      for (uint64_t loc_st = 0; loc_st < cur_loc.load(); loc_st += locs_per_batch) {
        uint64_t loc_ed = std::min(cur_loc.load(), loc_st + locs_per_batch);
        uint64_t st_sector = graph_loc_sector_no(loc_st);
        uint64_t ed_sector = graph_loc_sector_no(loc_ed > 0 ? loc_ed - 1 : 0);
        uint64_t n_sectors = ed_sector - st_sector + 1;
        std::vector<IORequest> reqs;
        reqs.push_back(IORequest(st_sector * SECTOR_LEN, n_sectors * SECTOR_LEN, buf, 0, 0));
        graph_reader_->read(reqs, ctx, false);

        for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
          if (loc >= loc2id_.size()) break;
          uint32_t id = loc2id_[loc];
          if (id == kInvalidID || id == kAllocatedID) continue;
          uint64_t loc_sector = graph_loc_sector_no(loc);
          auto page_buf = buf + (loc_sector - st_sector) * SECTOR_LEN;
          auto node = graph_node_from_page(page_buf, loc);
          uint64_t deg = node.nnbrs;
          total_degree += deg;
          total_nodes++;
          if (deg < min_deg) min_deg = deg;
          if (deg > max_deg) max_deg = deg;
          if (deg < params.R / 2) low_deg++;
          if (deg == 0) isolated++;
        }
      }
      aligned_free(buf);
    } else {
      void *ctx = reader->get_ctx();
      uint64_t sectors_per_batch = kBatchBytes / SECTOR_LEN;
      if (sectors_per_batch == 0) sectors_per_batch = 1;
      uint64_t locs_per_batch = sectors_per_batch * meta_.nnodes_per_sector;
      if (meta_.nnodes_per_sector == 0) locs_per_batch = 131072;
      char *buf = nullptr;
      alloc_aligned((void **) &buf, sectors_per_batch * SECTOR_LEN, SECTOR_LEN);

      for (uint64_t loc_st = 0; loc_st < cur_loc.load(); loc_st += locs_per_batch) {
        uint64_t loc_ed = std::min(cur_loc.load(), loc_st + locs_per_batch);
        uint64_t st_sector = loc_sector_no(loc_st);
        uint64_t ed_sector = loc_sector_no(loc_ed > 0 ? loc_ed - 1 : 0);
        uint64_t n_sectors = ed_sector - st_sector + 1;
        std::vector<IORequest> reqs;
        reqs.push_back(IORequest(st_sector * SECTOR_LEN, n_sectors * SECTOR_LEN, buf, 0, 0));
        reader->read(reqs, ctx, false);

        for (uint64_t loc = loc_st; loc < loc_ed; ++loc) {
          if (loc >= loc2id_.size()) break;
          uint32_t id = loc2id_[loc];
          if (id == kInvalidID || id == kAllocatedID) continue;
          uint64_t loc_sector = loc_sector_no(loc);
          auto page_buf = buf + (loc_sector - st_sector) * SECTOR_LEN;
          auto node = node_from_page(page_buf, loc);
          uint64_t deg = node.nnbrs;
          total_degree += deg;
          total_nodes++;
          if (deg < min_deg) min_deg = deg;
          if (deg > max_deg) max_deg = deg;
          if (deg < params.R / 2) low_deg++;
          if (deg == 0) isolated++;
        }
      }
      aligned_free(buf);
    }
    LOG(INFO) << "Graph stats: nodes=" << total_nodes
              << " avg_deg=" << (total_nodes > 0 ? (double)total_degree / total_nodes : 0)
              << " min_deg=" << min_deg << " max_deg=" << max_deg
              << " low_deg(<" << (params.R / 2) << ")=" << low_deg
              << " isolated=" << isolated;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
