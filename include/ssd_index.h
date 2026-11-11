#pragma once
#include <immintrin.h>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <set>
#include <unordered_map>
#include <omp.h>

#include "aligned_file_reader.h"
#include "ssd_index_defs.h"
#include "filter/selector.h"
#include "utils/concurrent_queue.h"
#include "utils/lock_table.h"
#include "utils/percentile_stats.h"
#include "utils/reader_epoch.h"
#include "mem_nav.h"
#include "nbr/nbr.h"
#include "utils.h"
#include "index.h"

enum SearchMode {
  BEAM_SEARCH = 0,
  PAGE_SEARCH = 1,
  PIPE_SEARCH = 2,
  CORO_SEARCH = 3,
  DECOUPLE_SEARCH = 4,
  DECOUPLE_BEAM_SEARCH = 5,
  DECOUPLE_PAGE_SEARCH = 6,
  STARLING_DECOUPLE_SEARCH = 7,  // decouple_search + _partition.bin.starling.decouple
  NON_DECOUPLE_SEARCH = 8
};

namespace pipeann {
inline constexpr const char kMemTopoIndexSuffix[] = "_mem_topo.index";

  class DecoupleVectorWriteCombiner {
   public:
    explicit DecoupleVectorWriteCombiner(pipeann::SparseLockTable<uint64_t> &lock_table)
        : vec_lock_table_(lock_table) {
    }

    void stage_write(uint64_t sector_no, uint64_t u_off, const void *point, uint64_t vec_data_len,
                     uint64_t size_per_io_vec, uint64_t nnodes_per_sector_vec, int vec_fd,
                     AlignedFileReader *vector_reader);
    // Persist one staged vector for graph publish visibility. Uses a small pwrite and
    // keeps the sector buffer so later inserts in the same sector avoid pread+full flush.
    void ensure_staged_vector_on_disk(uint64_t sector_no, uint64_t u_off, uint64_t vec_data_len, int vec_fd);
    // Read-side overlay lookup: if the vector at (sector_no, u_off) is still staged in
    // memory (not yet flushed to disk), copy vec_data_len bytes into dst and return true.
    // This decouples visibility from durability: searchers can resolve freshly inserted
    // vectors from the overlay, so inserts need not synchronously flush before publishing.
    bool try_read_staged_vector(uint64_t sector_no, uint64_t u_off, uint64_t vec_data_len, void *dst);
    // Fast check: are there any staged (un-flushed) vectors at all? Lets read-only
    // workloads skip all locking on the search hot path.
    inline bool has_pending() const { return pending_count_.load(std::memory_order_acquire) != 0; }
    // Flush one sector's full pending buffer to disk if present.
    void flush_sector(uint64_t sector_no, uint64_t size_per_io_vec, AlignedFileReader *vector_reader);
    void flush_all(uint64_t size_per_io_vec, AlignedFileReader *vector_reader);

   private:
    pipeann::SparseLockTable<uint64_t> &vec_lock_table_;
    struct PendingSector {
      char *buf;
      uint64_t num_writes = 0;
    };
    // Bound buffers even when reclaimed slots are scattered across many pages.
    static constexpr size_t kMaxPendingSectors = 64;
    std::unordered_map<uint64_t, PendingSector> pending_sector_bufs_;
    std::mutex pending_sector_bufs_mu_;
    std::atomic<uint64_t> pending_count_{0};
  };

  template<typename T, typename TagT = uint32_t>
  class SSDIndex {
   public:
    static constexpr uint32_t kAllocatedID = std::numeric_limits<uint32_t>::max() - 1;

    std::unique_ptr<MemNav> memnav_;
    void enable_memnav(uint32_t max_ids,uint32_t slots);
    std::vector<uint32_t> debug_memnav_seed_probe(const T *query,bool expect_fallback) {
      EpochGuard guard(reader_epoch_);
      auto *q=pop_query_buf(query);uint32_t seeds[32];float d[32];
      auto n=mem_search_ids_with_pq(query,q,32,seeds,d);
      push_query_buf(q);
      if(!n || (expect_fallback && (n!=1 || seeds[0]!=meta_.entry_point_id)))
        throw std::logic_error("MemNav seed probe failed");
      for(uint32_t i=0;i<n;++i)if(id2loc(seeds[i])==kInvalidID)
        throw std::logic_error("MemNav seed probe invalid mapping");
      return std::vector<uint32_t>(seeds,seeds+n);
    }
    std::unique_ptr<Index<T, uint32_t>> mem_index_;  // in-memory navigation graph
    // Cached reverse map for mem_topo tags -> live disk ids (invalidated when id2loc_ grows).
    mutable std::mutex tag2id_mu_;
    mutable std::unique_ptr<std::unordered_map<uint32_t, uint32_t>> tag2id_cache_;
    mutable size_t tag2id_cached_n_ = 0;

    // Index metadata (consolidated).
    SSDIndexMetadata<T> meta_;

    // For updates:
    // meta.npoints is the index's initial size (constant between two merges).
    // cur_id is the ID to be allocated (+1 for each insert, starting from meta.npoints)
    // cur_loc is the tail of the index file, which will be greater than cur_id if overprovisioned.
    // - This is because holes may exist in the index for update combining.
    std::atomic<uint64_t> cur_id, cur_loc;

    SSDIndex(pipeann::Metric m, std::shared_ptr<AlignedFileReader> &file_reader,
             AbstractNeighbor<T> *nbr = new PQNeighbor<T>(), bool tags = false,
             IndexBuildParameters *parameters = nullptr);

    ~SSDIndex();

    // Use node_from_page() to create instances for DiskNode.
    // Params: location (id2loc first if using ID), the page-aligned buffer.
    // The offset is calculated in the constructor.
    inline DiskNode<T> node_from_page(char *page_buf, uint32_t loc) {
      return DiskNode<T>(page_buf, loc, meta_);
    }

    // Size of the data region in a DiskNode.
    inline uint64_t node_label_size() const {
      return meta_.label_size;
    }

    // Unaligned offset to location.
    inline uint64_t u_loc_offset(uint64_t loc) {
      return loc * meta_.max_node_len;  // compacted store.
    }

    inline uint64_t u_loc_offset_nbr(uint64_t loc) {
      return loc * meta_.max_node_len + meta_.data_dim * sizeof(T);
    }

    // Avoid integer overflow when * SECTOR_LEN.
    inline uint64_t loc_sector_no(uint64_t loc) {
      return 1 + (meta_.nnodes_per_sector > 0 ? loc / meta_.nnodes_per_sector
                                              : loc * DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN));
    }

    inline uint64_t sector_to_loc(uint64_t sector_no, uint32_t sector_off) {
      return meta_.nnodes_per_sector == 0
                 ? (sector_no - 1) / DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN)  // sector_off == 0.
                 : (sector_no - 1) * meta_.nnodes_per_sector + sector_off;
    }

    void init_metadata(const SSDIndexMetadata<T> &meta) {
      meta.print();
      this->meta_ = meta;
      this->cur_id = this->cur_loc = meta.npoints;
      this->aligned_dim = ROUND_UP(meta_.data_dim, 8);
      this->params.R = meta.range;
      this->size_per_io = SECTOR_LEN * (meta_.nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(meta_.max_node_len, SECTOR_LEN));
      LOG(INFO) << "Size per IO: " << size_per_io;

      // Aligned.
      if (meta_.nnodes_per_sector != 0 && meta_.npoints % meta_.nnodes_per_sector != 0) {
        cur_loc += meta_.nnodes_per_sector - (meta_.npoints % meta_.nnodes_per_sector);
      }
      LOG(INFO) << "Cur location: " << this->cur_loc;

      // Update-related metadata, if not initialized in constructor, initialize here.
      if (params.L == 0) {
        // Experience values.
        LOG(INFO) << "Automatically set the update-related parameters.";
        params.set(meta.range, meta.range + 32, 384, 1.2, 0, true, 4);
        params.print();
      }
    }

    void init_query_buf(QueryBuffer<T> &buf) {
      pipeann::alloc_aligned((void **) &buf.coord_scratch, this->aligned_dim * sizeof(T), 8 * sizeof(T));
      uint64_t sector_scratch_sz = decouple_mode_
                                       ? std::max(size_per_io_graph_, size_per_io_vec_)
                                       : size_per_io;
      pipeann::alloc_aligned((void **) &buf.sector_scratch, MAX_N_SECTOR_READS * sector_scratch_sz, SECTOR_LEN);
      pipeann::alloc_aligned((void **) &buf.nbr_vec_scratch,
                             MAX_N_EDGES * AbstractNeighbor<T>::MAX_BYTES_PER_NBR * sizeof(uint8_t), 256);
      pipeann::alloc_aligned((void **) &buf.nbr_ctx_scratch, ROUND_UP(nbr_handler->query_ctx_size(), 256), 256);
      pipeann::alloc_aligned((void **) &buf.aligned_dist_scratch, MAX_N_EDGES * sizeof(float), 256);
      pipeann::alloc_aligned((void **) &buf.aligned_query_T, this->aligned_dim * sizeof(T), 8 * sizeof(T));
      // Per-thread vector write buffer (one sector).
      if (buf.vec_write_buf == nullptr) {
        pipeann::alloc_aligned((void **) &buf.vec_write_buf, size_per_io_vec_, SECTOR_LEN);
      }

      buf.visited = new tsl::robin_set<uint64_t>(4096);
      buf.page_visited = new tsl::robin_set<unsigned>(4096);

      memset(buf.sector_scratch, 0, MAX_N_SECTOR_READS * sector_scratch_sz);
      memset(buf.coord_scratch, 0, this->aligned_dim * sizeof(T));
      memset(buf.aligned_query_T, 0, this->aligned_dim * sizeof(T));
    }

    QueryBuffer<T> *pop_query_buf(const T *query) {
      QueryBuffer<T> *data = this->thread_data_queue.pop();
      while (data == this->thread_data_queue.null_T) {
        this->thread_data_queue.wait_for_push_notify();
        data = this->thread_data_queue.pop();
      }

      if (likely(query != nullptr)) {
        if (this->metric == Metric::COSINE) {
          // Data has been normalized. Normalize search vector too.
          pipeann::normalize_data(data->aligned_query_T, query, meta_.data_dim);
        } else {
          memcpy(data->aligned_query_T, query, meta_.data_dim * sizeof(T));
        }
      }
      return data;
    }

    void push_query_buf(QueryBuffer<T> *data) {
      this->thread_data_queue.push(data);
      this->thread_data_queue.push_notify_all();
    }

    // Load compressed data, and obtains the handle to the disk-resident index.
    int load(const char *index_prefix, uint32_t num_threads, bool use_page_search = false);

    void load_mem_index(const std::string &mem_index_path);

    // Load disk index to memory index.
    Index<T, TagT> *load_to_mem(const std::string &filename);

    // Search supporting update.
    size_t beam_search(const T *query, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT *res_tags, float *res_dists, const uint64_t beam_width, QueryStats *stats = nullptr,
                       tsl::robin_set<uint32_t> *deleted_nodes = nullptr, bool dyn_search_l = true);

    size_t coro_search(T **queries, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT **res_tags, float **res_dists, const uint64_t beam_width, int N);

    // Read-only search algorithms.
    size_t page_search(const T *query, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT *res_tags, float *res_dists, const uint64_t beam_width, QueryStats *stats = nullptr);

    size_t pipe_search(const T *query, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT *res_tags, float *res_dists, const uint64_t beam_width, QueryStats *stats = nullptr,
                       AbstractSelector *selector = nullptr, const void *filter_data = nullptr,
                       const uint64_t relaxed_monotonicity_l = 0);

    // Decoupled search: graph and vectors stored separately.
    // Graph file (_disk.graph): topology only (neighbors), page-aligned, no sector-0 metadata.
    // Vector file (_disk.vector): flat exact vectors, id == loc, page-aligned, no metadata.
    // Metadata file (_disk.index): single 4K sector with SSDIndexMetadata.
    // partition_suffix selects page layout, e.g. "dgai" or "starling.decouple".
    // When suffix != "dgai", requires _disk.graph.<suffix> (no fallback to _disk.graph).
    int load_decouple_index(const char *index_prefix, uint32_t num_threads,
                            const std::string &partition_suffix = "dgai");

    struct InsertContext {
      tsl::robin_map<uint32_t, T *> coord_map;
      T *coord_buf;
      std::vector<uint64_t> hint_pages;
      std::vector<uint64_t> page_ref;

      InsertContext(uint64_t max_nodes, uint64_t aligned_dim) {
        coord_map.reserve(max_nodes);
        pipeann::alloc_aligned((void **) &coord_buf, max_nodes * aligned_dim * sizeof(T), 256);
      }

      ~InsertContext() {
        pipeann::aligned_free(coord_buf);
      }

      InsertContext(const InsertContext &) = delete;
      InsertContext &operator=(const InsertContext &) = delete;
      InsertContext(InsertContext &&) = delete;
      InsertContext &operator=(InsertContext &&) = delete;
    };

    // Decoupled beam search for insert: same interface as do_beam_search.
    // Reads graph from graph_reader_, vectors from vector_reader_.
    void do_decouple_search(const T *vec, uint32_t mem_L, uint32_t Lsize, const uint32_t beam_width,
                            std::vector<Neighbor> &expanded_nodes_info,
                            InsertContext *insert_ctx = nullptr, QueryStats *stats = nullptr);

    // Synchronous decoupled beam search – identical interface to do_beam_search.
    // Uses graph_reader_/vector_reader_ and DiskGraphNode (no coords in graph).
    // Phase 2 re-reads exact vectors for final distances.
    void do_decouple_beam_search(const T *query1, uint32_t mem_L, uint32_t l_search, const uint32_t beam_width,
                                  std::vector<Neighbor> &expanded_nodes_info,
                                  tsl::robin_map<uint32_t, T *> *coord_map = nullptr,
                                  T *coord_buf = nullptr, QueryStats *stats = nullptr,
                                  tsl::robin_set<uint32_t> *exclude_nodes = nullptr, bool dyn_search_l = true,
                                  std::vector<uint64_t> *passthrough_page_ref = nullptr);

    // Full decoupled search with optional top-K tag/distance output for query path.
    // When insert_ctx is non-null, poll_alloc records page cache refs in
    // insert_ctx->page_ref (for insert read_alloc reuse) and fills coord_map.
    // Exact-vector Phase 2 reranks a window of size rerank_window centered on K:
    //   [max(0, K - W/2), min(L, K + W/2)). Insert forces start=0 so prune sees exact coords.
    // NOTE: callers that manage their own query buffer (e.g. insert) must
    // push_query_buf themselves; decouple_search always pushes it on return.
    size_t decouple_search(const T *query, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                           std::vector<Neighbor> &expanded_nodes_info,
                           const uint64_t beam_width, QueryStats *stats = nullptr,
                           TagT *res_tags = nullptr, float *res_dists = nullptr,
                           uint64_t rerank_window = 30,
                           InsertContext *insert_ctx = nullptr);

    // Page-based decoupled search: reads full graph pages and expands all
    // co-resident nodes (deferred page processing), then reranks with exact
    // vectors in Phase 2.  Returns the number of results written to res_tags.
    size_t decouple_page_search(const T *query1, const uint64_t k_search, const uint32_t mem_L,
                                const uint64_t l_search, TagT *res_tags, float *distances,
                                const uint64_t beam_width, QueryStats *stats = nullptr);

    // Non-decoupled single-stage search: reads full pages (graph + coords) via
    // a single reader. Exact distances are computed during traversal — no Phase 2.
    // Works on regular (non-decoupled) indexes loaded via load().
    size_t non_decouple_search(const T *query, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                               std::vector<Neighbor> &expanded_nodes_info,
                               const uint64_t beam_width, QueryStats *stats = nullptr,
                               tsl::robin_map<uint32_t, T *> *coord_map = nullptr,
                               T *coord_buf = nullptr,
                               TagT *res_tags = nullptr, float *res_dists = nullptr,
                               std::vector<uint64_t> *passthrough_page_ref = nullptr,
                               uint64_t max_candidates = 30);

    int insert_in_place(const T *point, const TagT &tag, tsl::robin_set<uint32_t> *deletion_set = nullptr,
                        InsertStats *stats = nullptr);

    // Decoupled insert: graph and vectors stored separately.
    // search_mode controls the internal search algorithm:
    //   DECOUPLE_SEARCH (4)      → uses decouple_search (supports coord_map / exact rerank)
    //   DECOUPLE_BEAM_SEARCH (5) → uses do_decouple_beam_search (original)
    int decouple_insert_in_place(const T *point, const TagT &tag, tsl::robin_set<uint32_t> *deletion_set = nullptr,
                                 InsertStats *stats = nullptr,
                                 int search_mode = DECOUPLE_BEAM_SEARCH);

    // Merge deletes (NOTE: index read-only during merge.)
    void merge_deletes(const std::string &in_path_prefix, const std::string &out_path_prefix,
                       const std::vector<TagT> &deleted_nodes, const tsl::robin_set<TagT> &deleted_nodes_set,
                       uint32_t nthreads, const uint32_t &n_sampled_nbrs);

    struct TopologyCompactionStats {
      uint64_t sources_attempted = 0;
      uint64_t sources_merged = 0;
      uint64_t pages_reclaimed = 0;
      uint64_t page_reads = 0;
      uint64_t page_writes = 0;
      uint64_t two_hop_cache_hits = 0;
      uint64_t resident_hits = 0;
      uint64_t moved_nodes = 0;
      uint64_t exact_added_internal_edges = 0;
      double elapsed_us = 0;
    };
    // Caller holds the exclusive update/merge locks and has drained background
    // I/O. Relocates topology records only; internal IDs and vectors stay fixed.
    // Optional resident pages contain the repaired final deletion batch. Writes
    // update this range too; it does not expose extra pages to two-hop probing.
    TopologyCompactionStats compact_topology_pages(char *resident_ptr = nullptr, uint64_t first_page = 0,
                                                   uint64_t page_count = 0);

    // Repair deleted edges in-place, then coalesce sparse topology pages.
    void decouple_merge_deletes(const std::string &in_path_prefix, const std::string &out_path_prefix,
                                const std::vector<TagT> &deleted_nodes,
                                const tsl::robin_set<TagT> &deleted_nodes_set, uint32_t nthreads,
                                const uint32_t &n_sampled_nbrs);

    // After merge, reload the index.
    void reload(const char *index_prefix, uint32_t num_threads);

    void write_metadata_and_pq(const std::string &in_path_prefix, const std::string &out_path_prefix,
                               std::vector<TagT> *new_tags = nullptr);

    void copy_index(const std::string &prefix_in, const std::string &prefix_out,
                    int search_mode = BEAM_SEARCH);

    // Background insert I/O commit.
    struct BgTask {
      QueryBuffer<T> *thread_data;
      std::vector<IORequest> writes;
      std::vector<uint64_t> pages_to_unlock;
      std::vector<uint64_t> pages_to_deref;
      bool terminate = false;
    };

    using PageArr = std::vector<uint32_t>;

    static constexpr int kBgIOThreads = 1;

    // Derived/runtime metadata not stored in SSDIndexMetadata.
    uint64_t aligned_dim = 0;
    uint64_t size_per_io = 0;

    // File reader and index file path.
    std::shared_ptr<AlignedFileReader> &reader;
    std::string disk_index_file;

    // Neighbor handler.
    AbstractNeighbor<T> *nbr_handler;

    // Distance comparator.
    pipeann::Metric metric;
    std::shared_ptr<Distance<T>> dist_cmp;

    // Update-related parameters.
    IndexBuildParameters params;

    // Thread-specific scratch buffers.
    ConcurrentQueue<QueryBuffer<T> *> thread_data_queue;
    std::vector<QueryBuffer<T> *> thread_data_bufs;  // pre-allocated thread data
    uint64_t max_nthreads;

    // Background I/O threads for insert.
    ConcurrentQueue<BgTask *> bg_tasks = ConcurrentQueue<BgTask *>(nullptr);
    std::thread *bg_io_thread_[kBgIOThreads]{nullptr};

    // Locking tables for concurrency control.
    pipeann::SparseLockTable<uint64_t> page_lock_table{"page"};
    pipeann::SparseLockTable<uint64_t> vec_lock_table{"vec"};
    pipeann::SparseLockTable<uint64_t> page_idx_lock_table{"page_idx"};
    pipeann::SparseLockTable<uint64_t> idx_lock_table{"idx"};
    std::shared_mutex merge_lock;  // serve search during merge.
    // Set before taking exclusive merge_lock so online load stops admitting new
    // shared_lock holders (avoids writer starvation under continuous search).
    std::atomic<bool> merge_remap_exclusive_{false};
    // Legacy flag (page-lock merge coord). Unused under seqlock+EBR path; kept
    // so exclusive-merge A/B builds and older binaries remain ABI-compatible.
    std::atomic<bool> page_merge_coord_{false};

    // Page seqlock versions + reader epoch for search∥merge (see reader_epoch.h).
    // Odd = write in progress; even = stable. Search samples before IO and
    // re-reads on torn/mismatch. Merge calls synchronize_epoch before reclaim.
    std::atomic<uint32_t> *page_ver_ = nullptr;
    uint64_t page_ver_cap_ = 0;
    std::mutex page_ver_mu_;
    // Prior page_ver_ arrays kept until ~SSDIndex so concurrent page_ver_load
    // cannot UAF when the table grows under online search.
    std::vector<std::atomic<uint32_t> *> page_ver_retired_;
    ReaderEpoch reader_epoch_;

    void ensure_page_versions(uint64_t n_pages);
    void force_page_versions_even();
    uint32_t page_ver_load(uint64_t page) const {
      auto *table = page_ver_;
      const uint64_t cap = page_ver_cap_;
      if (table == nullptr || page >= cap) return 0;
      return table[page].load(std::memory_order_acquire);
    }
    void begin_page_write(uint64_t page) {
      ensure_page_versions(page + 1);
      // Force odd even if a prior mismatched begin left an odd value.
      uint32_t v = page_ver_[page].load(std::memory_order_relaxed);
      if ((v & 1) == 0) page_ver_[page].fetch_add(1, std::memory_order_acq_rel);
    }
    void end_page_write(uint64_t page) {
      if (page >= page_ver_cap_ || page_ver_ == nullptr) return;
      uint32_t v = page_ver_[page].load(std::memory_order_relaxed);
      if (v & 1) page_ver_[page].fetch_add(1, std::memory_order_release);
    }
    void begin_page_writes(uint64_t first_page, uint64_t n_pages) {
      ensure_page_versions(first_page + n_pages);
      for (uint64_t i = 0; i < n_pages; ++i) begin_page_write(first_page + i);
    }
    void end_page_writes(uint64_t first_page, uint64_t n_pages) {
      for (uint64_t i = 0; i < n_pages; ++i) end_page_write(first_page + i);
    }
    // Pause OnlineSearchLoad admissions while draining epochs so G1/G2 cannot
    // race a continuous stream of new EpochGuards / stuck readers forever.
    void synchronize_epoch() {
      const bool already = merge_remap_exclusive_.exchange(true, std::memory_order_acq_rel);
      reader_epoch_.synchronize(/*timeout_ms=*/2000);
      if (!already) merge_remap_exclusive_.store(false, std::memory_order_release);
    }

    // Page search mode flag.
    bool use_page_search_ = true;

    // ID to location mapping.
    // Concurrency control is done in lock_idx.
    // Only resize should be protected.
    std::vector<uint32_t> id2loc_;
    pipeann::ReaderOptSharedMutex id2loc_resize_mu_;

    // Location to ID mapping.
    // If nnodes_per_sector >= 1, page_layout[i * nnodes_per_sector + j] is the id of the j-th node in the i-th page.
    // ElseIf nnodes_per_sector == 0, page_layout[i] is the id of the i-th node (starting from loc_sector_no(i)).
    std::vector<uint32_t> loc2id_;
    pipeann::ReaderOptSharedMutex loc2id_resize_mu_;
    std::mutex alloc_lock;
    ConcurrentQueue<uint32_t> empty_pages = ConcurrentQueue<uint32_t>(kInvalidID);
    std::atomic<uint64_t> reused_topology_pages_{0};

    // Topology split planning runs after the source page has been read.
    uint64_t loaded_empty_topology_pages_ = 0;
    struct OnlineSplitEvent {
      double allocator_us = 0, allocator_wait_us = 0;
      double plan_us = 0, plan_wait_us = 0, topology_snapshot_us = 0;
      double pq_distance_us = 0, partition_us = 0;
      uint64_t pq_pairs = 0, pq_batches = 0, local_directed_edges = 0;
      uint64_t zero_degree_nodes = 0, no_edge_fallbacks = 0;
      uint64_t initializations = 0, sweeps = 0, refinement_moves = 0;
      uint64_t old_page_size = 0, new_page_size = 0, moved_old_nodes = 0;
      bool target_in_new_page = false;
    };
    struct OnlineSplitWindow {
      OnlineSplitEvent totals;
      uint64_t split_count = 0;
      double max_us = 0;
      // Ring of the latest samples: bounded even if no caller drains stats.
      std::vector<double> plan_times;
      std::vector<uint64_t> smaller_size_hist, old_size_hist, new_size_hist;
      uint64_t zero_old_victim_splits = 0, target_in_new_page_splits = 0;
    } online_split_window_;
    static constexpr size_t kOnlineSplitTimingSamples = 4096;
    std::mutex online_split_stats_mutex_;
    uint64_t online_split_drain_sequence_ = 0, online_split_cumulative_count_ = 0;
    uint64_t online_split_cumulative_pq_pairs_ = 0;
    double online_split_cumulative_plan_us_ = 0;

    // Call after joining an insertion batch. Returns and resets window counters;
    // cumulative counters are retained. Percentiles use at most the latest
    // 4096 timings; counts, totals, maxima and histograms remain exact.
    // No logging or file I/O occurs per split.
    std::string take_online_split_stats_json();
    void record_online_split_event(const OnlineSplitEvent &event);
    // Load/exclusive maintenance only: rebuild the free-page queue from the
    // already loaded mapping, without reading or truncating the graph file.
    void rebuild_decouple_empty_pages();

    // IDs become reusable only after an exclusive deletion merge has repaired
    // every incoming edge and drained old readers. Protected by alloc_lock.
    // Descending order makes pop_back() consume slots from the same vector page.
    std::vector<uint32_t> free_decouple_ids_;
    std::unordered_map<uint64_t, uint64_t> recycled_vector_writes_left_;
    std::atomic<uint64_t> reused_vector_slots_{0};

  // Tag support.
    // If ID == tag, then it is not stored.
    libcuckoo::cuckoohash_map<uint32_t, TagT> tags;

    // Flags.
    bool load_flag = false;    // already loaded.
    bool enable_tags = false;  // support for tags and dynamic indexing

    // Decoupled index support.
    bool decouple_mode_ = false;
    // A deleted initial entry remains a routing-only node; its original
    // ID/PQ/vector slot is never released for reuse.
    uint32_t frozen_entry_id_ = kInvalidID;
    std::atomic<uint32_t> routing_only_entry_id_{kInvalidID};
    inline bool is_routing_only(uint32_t id) const {
      return id != kInvalidID && id == routing_only_entry_id_.load(std::memory_order_acquire);
    }
    std::shared_ptr<AlignedFileReader> graph_reader_;
    std::shared_ptr<AlignedFileReader> vector_reader_;
    std::string graph_file_;
    std::string vector_file_;
    std::mutex decouple_insert_mutex_;  // serialize decoupled inserts

    // Write-combining buffer for decoupled vector file appends.
    DecoupleVectorWriteCombiner vec_write_combiner_{vec_lock_table};

    // Graph-specific layout (differs from original because records have no coords).
    uint64_t max_graph_node_len_ = 0;
    uint64_t nnodes_per_sector_graph_ = 0;
    uint64_t size_per_io_graph_ = 0;

    // Vector-specific layout: same packed-record pattern as original, no sector 0.
    uint64_t nnodes_per_sector_vec_ = 0;
    uint64_t size_per_io_vec_ = 0;

    // Decoupled vector write-combining helpers.
    // stage_decouple_vector_write() merges writes into a per-sector buffer and
    // flushes on full-sector boundaries.
    void stage_decouple_vector_write(uint32_t id, const T *point, bool reused_id = false);
    // Ensure target vector is on disk before graph publication (no search-side overlay).
    void ensure_decouple_vector_on_disk(uint32_t id);
    // Read-side overlay lookup used by decoupled search Phase-2 reranking. Returns true
    // (and fills dst with the exact vector) if id's vector is still staged in memory.
    // Lets visibility be decoupled from durability so inserts skip the synchronous flush.
    bool try_read_decouple_vector(uint32_t id, void *dst);
    // Flush all pending vector buffers.
    void flush_all_pending_decouple_vectors();

    // Graph file: no sector 0 metadata, records are [nnbrs | nbrs | labels] only.
    inline uint64_t graph_loc_sector_no(uint64_t loc) const {
      return nnodes_per_sector_graph_ > 0
                 ? loc / nnodes_per_sector_graph_
                 : loc * DIV_ROUND_UP(max_graph_node_len_, SECTOR_LEN);
    }

    inline uint64_t graph_u_loc_offset(uint64_t loc) {
      return loc * max_graph_node_len_;
    }

    inline DiskGraphNode<T> graph_node_from_page(char *page_buf, uint32_t loc, uint32_t node_id = kInvalidID,
                                                const char *site = "graph_node_from_page") {
      DiskGraphNode<T> node(page_buf, loc, max_graph_node_len_, nnodes_per_sector_graph_, meta_.range);
#ifdef DGAI_STRICT_GRAPH_CHECKS
      check_graph_nnbrs(node.nnbrs, node_id, loc, site);
#endif
      return node;
    }

    // Vector file: packed records (same pattern as original), no sector 0.
    // id == loc in ordering; physical offset follows sector packing.
    inline uint64_t vec_loc_sector_no(uint64_t id) {
      uint64_t vec_len = meta_.data_dim * sizeof(T);
      return nnodes_per_sector_vec_ > 0
                 ? id / nnodes_per_sector_vec_
                 : id * DIV_ROUND_UP(vec_len, SECTOR_LEN);
    }

    // Byte offset of vector id within its sector.
    inline uint64_t vec_sector_offset(uint64_t id) {
      return nnodes_per_sector_vec_ > 0
                 ? (id % nnodes_per_sector_vec_) * meta_.data_dim * sizeof(T)
                 : 0;
    }

    void init_buffers(uint64_t nthreads);
    void destroy_buffers();

    // Load id2loc and loc2id (i.e., page_layout), to support index reordering.
    // suffix: ".aligned" (default, fallback .starling) or ".dgai" (decouple mode).
    void load_page_layout(const std::string &index_prefix, const uint64_t nnodes_per_sector = 0,
                          const uint64_t num_points = 0, const std::string &suffix = "");

    void load_tags(const std::string &tag_file, size_t offset = 0);

    // Direct insert related.
    void do_beam_search(const T *vec, uint32_t mem_L, uint32_t Lsize, const uint32_t beam_width,
                        std::vector<Neighbor> &expanded_nodes_info, tsl::robin_map<uint32_t, T *> *coord_map = nullptr,
                        T *coord_buf = nullptr, QueryStats *stats = nullptr,
                        tsl::robin_set<uint32_t> *exclude_nodes = nullptr, bool dyn_search_l = true,
                        std::vector<uint64_t> *passthrough_page_ref = nullptr);

    // Background I/O thread function.
    void bg_io_thread();

    int get_vector_by_id(const uint32_t &id, T *vector);

    // ID, loc, page mapping.
    TagT id2tag(uint32_t id);

    uint32_t id2loc(uint32_t id);
    void set_id2loc(uint32_t id, uint32_t loc);
    uint64_t id2page(uint32_t id);

    uint32_t loc2id(uint32_t loc);
    void set_loc2id(uint32_t loc, uint32_t id);
    void erase_loc2id(uint32_t loc);

    PageArr get_page_layout(uint32_t page_no);

    void erase_and_set_loc(const std::vector<uint64_t> &old_locs, const std::vector<uint64_t> &new_locs,
                           const std::vector<uint32_t> &new_ids);
    // Returns <loc, need_read>.
    std::vector<uint64_t> alloc_loc(int n, const std::vector<uint64_t> &hint_pages,
                                    std::set<uint64_t> &page_need_to_read);

    // Decouple graph alloc (under alloc_lock). locs[new_nhood.size()] is the target.
    // Returns true when a split is required. In that case split_source_page is
    // the old page and locs.back() reserves a new page; no partition is chosen
    // until the caller reads the source topology and calls plan_pq_graph_split.
    // With -DDGAI_DISABLE_GRAPH_SPLIT, never splits: empty hint slots still reused,
    // otherwise the target is always allocated at the graph file tail.
    bool alloc_graph_loc(uint32_t target_id, const std::vector<uint32_t> &new_nhood,
                         const std::vector<Neighbor> &placement_neighbors,
                         std::set<uint64_t> &pages_need_to_read, std::vector<uint64_t> &locs,
                         uint64_t &split_source_page, double *alloc_lock_wait_us = nullptr,
                         uint64_t *recycled_page = nullptr);

    // Reuses the page already reserved by alloc_graph_loc. The caller holds
    // both source/destination page locks and has finished reading page_buffers.
    void plan_pq_graph_split(uint32_t target_id, uint64_t source_page,
                              const std::vector<uint32_t> &new_nhood,
                              const std::unordered_map<uint64_t, char *> &page_buffers,
                              QueryBuffer<T> *scratch, std::vector<uint64_t> &locs,
                              std::vector<uint64_t> &split_orig_locs,
                              std::vector<uint64_t> &split_new_locs,
                              std::vector<uint32_t> &split_ids, OnlineSplitEvent &event);

   public:
    void verify_id2loc();
    // Scan on-disk graph: every neighbor id must be < cur_id and have a valid id2loc.
    void verify_graph_neighbor_ids();
    void log_graph_stats();  // debug: log degree distribution

   private:
    inline void check_graph_loc(uint32_t node_id, uint32_t loc, const char *site) const {
#ifdef DGAI_STRICT_GRAPH_CHECKS
      if (unlikely(loc == kInvalidID)) {
        LOG(ERROR) << "Invalid graph loc (kInvalidID): site=" << (site != nullptr ? site : "unknown")
                   << " node_id=" << node_id << " cur_id=" << cur_id.load();
        crash();
      }
#else
      (void) node_id;
      (void) loc;
      (void) site;
#endif
    }

    inline void check_graph_node_mapping(uint32_t node_id, uint32_t loc, const char *site) const {
#ifdef DGAI_STRICT_GRAPH_CHECKS
      if (unlikely(node_id >= cur_id.load())) {
        LOG(ERROR) << "Invalid graph node id >= cur_id: site=" << (site != nullptr ? site : "unknown")
                   << " node_id=" << node_id << " loc=" << loc << " cur_id=" << cur_id.load();
        crash();
      }
      check_graph_loc(node_id, loc, site);
#else
      (void) node_id;
      (void) loc;
      (void) site;
#endif
    }

    inline void check_graph_loc_stable(uint32_t node_id, uint32_t loc, uint32_t expected_loc,
                                     const char *site) const {
#ifdef DGAI_STRICT_GRAPH_CHECKS
      check_graph_node_mapping(node_id, loc, site);
      if (unlikely(loc != expected_loc)) {
        LOG(ERROR) << "Stale graph loc: site=" << (site != nullptr ? site : "unknown")
                   << " node_id=" << node_id << " expected_loc=" << expected_loc << " id2loc=" << loc
                   << " cur_id=" << cur_id.load();
        crash();
      }
#else
      (void) node_id;
      (void) loc;
      (void) expected_loc;
      (void) site;
#endif
    }

    inline void check_graph_nnbrs(uint32_t nnbrs, uint32_t node_id, uint32_t loc, const char *site) const {
#ifdef DGAI_STRICT_GRAPH_CHECKS
      check_graph_loc(node_id, loc, site);
      if (unlikely(nnbrs > meta_.range)) {
        LOG(ERROR) << "Invalid graph nnbrs: nnbrs=" << nnbrs << " max_range=" << meta_.range
                   << " site=" << (site != nullptr ? site : "unknown") << " node_id=" << node_id << " loc=" << loc
                   << " page=" << graph_loc_sector_no(loc) << " cur_id=" << cur_id.load();
        crash();
      }
#else
      (void) nnbrs;
      (void) node_id;
      (void) loc;
      (void) site;
#endif
    }

    // lock-related.
    void lock_vec(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                  const std::vector<uint32_t> &neighbors, bool rd = false);
    void unlock_vec(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                    const std::vector<uint32_t> &neighbors);

    std::vector<uint32_t> get_to_lock_idx(uint32_t target, const std::vector<uint32_t> &neighbors);

    // Lock the mapping for target/page if use_page_search == false/true.
    std::vector<uint32_t> lock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                                   const std::vector<uint32_t> &neighbors, bool rd = false);
    void unlock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, const std::vector<uint32_t> &to_lock);
    void unlock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, const uint32_t &to_lock);

    // Two-level lock, as id2page may change before and after grabbing the lock.
    std::vector<uint32_t> lock_page_idx(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                                        const std::vector<uint32_t> &neighbors, bool rd = false);
    void unlock_page_idx(pipeann::SparseLockTable<uint64_t> &lock_table, const std::vector<uint32_t> &to_lock);

    // Always returns live internal IDs, for both MemNav and legacy tagged topology.
    // Legacy PQ lookup resolves tags before reading the disk PQ table.
    uint32_t mem_search_ids_with_pq(const T *query, QueryBuffer<T> *query_buf, uint32_t mem_L, uint32_t *ids_out,
                                float *dists_out);

    // mem_topo returns external tags; disk PQ / beam frontier need internal ids once tag!=id.
    // Caller must hold tag2id_mu_ through the returned-map lookup.
    const std::unordered_map<uint32_t, uint32_t> &tag2id_map();
    bool tag_to_live_id(uint32_t tag, uint32_t &out_id);
    // Compact tags (+ optional aligned dists) into live internal ids. Returns count kept.
    uint32_t resolve_mem_tags_to_ids(const uint32_t *tags, uint32_t n, uint32_t *ids_out,
                                     float *dists_inout = nullptr);
  };
}  // namespace pipeann
