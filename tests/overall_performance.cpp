#include "dynamic_index.h"

#include <index.h>
#include <cstddef>
#include <future>
#include <numeric>
#include <omp.h>
#include <string.h>
#include <time.h>
#include "utils/timer.h"
#include <cstring>
#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <dirent.h>
#include <sys/stat.h>

#include "index.h"
#include "utils/kmeans_utils.h"
#include "utils/partition.h"
#include "utils.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <chrono>

#define MERGE_ROUND 1
#define MERGE_IO_THRESHOLD 1.2
#define NUM_INSERT_THREADS 10
#define NUM_MERGE_THREADS 20
#define NUM_DELETE_THREADS 1
#ifndef NUM_SEARCH_THREADS
#define NUM_SEARCH_THREADS 16
#endif
#define DeleteQPS 1000

int begin_time = 0;
pipeann::Timer globalTimer;

// acutually also shows disk size
void ShowMemoryStatus(const std::string &filename) {
  int current_time = globalTimer.elapsed() / 1.0e6f - begin_time;

  int tSize = 0, resident = 0, share = 0;
  std::ifstream buffer("/proc/self/statm");
  buffer >> tSize >> resident >> share;
  buffer.close();
  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024;  // in case x86-64 is configured to use 2MB pages
  double rss = resident * page_size_kb;

  struct stat st;
  memset(&st, 0, sizeof(struct stat));
  std::string index_file_name = filename + "_disk.index";
  stat(index_file_name.c_str(), &st);

  std::cout << " memory current time: " << current_time << " RSS : " << rss << " KB " << index_file_name
            << " Index size " << (st.st_size / (1 << 20)) << " MB" << std::endl;
}

std::string convertFloatToString(const float value, const int precision = 0) {
  std::stringstream stream{};
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

std::string GetTruthFileName(std::string &truthFilePrefix, int l_start) {
  std::string fileName(truthFilePrefix);
  fileName = fileName + "/gt_" + std::to_string(l_start) + ".bin";
  LOG(INFO) << "Truth file name: " << fileName;
  return fileName;
}

template<typename T, typename TagT>
float sync_search_kernel(T *query, size_t query_num, size_t query_dim, const int recall_at, uint64_t L,
                        uint32_t beam_width, pipeann::DynamicSSDIndex<T, TagT> &sync_index, std::string &truthset_file,
                        bool merged, bool calRecall, double &disk_io) {
  unsigned *gt_ids = NULL;
  float *gt_dists = NULL;
  size_t gt_num, gt_dim;

  if (!file_exists(truthset_file)) {
    LOG(WARNING) << "Truthset missing: " << truthset_file
                 << " — recall will be reported as 0 (not computed).";
    calRecall = false;
  }

  if (calRecall) {
    std::cout << "current truthfile: " << truthset_file << std::endl;
    pipeann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
  }

  float *query_result_dists = new float[recall_at * query_num];
  TagT *query_result_tags = new TagT[recall_at * query_num];

  for (uint32_t q = 0; q < query_num; q++) {
    for (uint32_t r = 0; r < (uint32_t) recall_at; r++) {
      query_result_tags[q * recall_at + r] = std::numeric_limits<TagT>::max();
      query_result_dists[q * recall_at + r] = std::numeric_limits<float>::max();
    }
  }

  std::vector<double> latency_stats(query_num, 0);
  pipeann::QueryStats *stats = new pipeann::QueryStats[query_num];
  std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS " << std::setw(18) << "Mean Lat" << std::setw(12)
            << "50 Lat" << std::setw(12) << "90 Lat" << std::setw(12) << "95 Lat" << std::setw(12) << "99 Lat"
            << std::setw(12) << "99.9 Lat" << std::setw(12) << recall_string << std::setw(12) << "Disk IOs"
            << std::endl;
  std::cout << "==============================================================="
               "==============="
            << std::endl;
  auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(NUM_SEARCH_THREADS) schedule(dynamic)
  for (int64_t i = 0; i < (int64_t) query_num; i++) {
    auto qs = std::chrono::high_resolution_clock::now();
    // stats[i].n_current_used = 8;
    sync_index.search(query + i * query_dim, recall_at, 0, L, beam_width, query_result_tags + i * recall_at,
                      query_result_dists + i * recall_at, stats + i, true);

    auto qe = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = qe - qs;
    latency_stats[i] = diff.count() * 1000;
    // std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  auto e = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = e - s;
  float qps = (query_num / diff.count());
  float recall = 0;

  int current_time = globalTimer.elapsed() / 1.0e6f - begin_time;
  if (calRecall) {
    recall = pipeann::calculate_recall(query_num, gt_ids, gt_dists, gt_dim, query_result_tags, recall_at, recall_at);
    delete[] gt_ids;
  }

  std::cout << "search current time: " << current_time << std::endl;

  float mean_ios =
      (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.n_ios; });

  std::sort(latency_stats.begin(), latency_stats.end());
  std::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
            << ((float) std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0f)) / (float) query_num
            << std::setw(12) << (float) latency_stats[(uint64_t) (0.50 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(uint64_t) (0.90 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(uint64_t) (0.95 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(uint64_t) (0.99 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(uint64_t) (0.999 * ((double) query_num))] << std::setw(12) << recall
            << std::setw(12) << mean_ios << std::endl;
  disk_io = mean_ios;

  delete[] query_result_dists;
  delete[] query_result_tags;
  delete[] stats;
  return qps;
}

// Continuous online search load during insert/delete/merge.
// Reports both input_qps (starts/wall) and completed_qps (finishes/wall).
// Table "QPS" column = completed throughput (same meaning as sync_search_kernel).
template<typename T, typename TagT>
struct OnlineSearchLoad {
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> started{0};
  std::atomic<uint64_t> completed{0};
  std::mutex lat_mu;
  std::vector<double> lat_ms;
  std::vector<std::thread> workers;
  std::chrono::high_resolution_clock::time_point t0;
  std::chrono::high_resolution_clock::time_point last_snap_t;
  uint64_t last_snap_completed{0};
  uint64_t last_snap_started{0};
  std::string phase;

  void start(T *query, size_t query_num, size_t query_dim, int recall_at, uint64_t L, uint32_t beam_width,
             pipeann::DynamicSSDIndex<T, TagT> &sync_index, const std::string &phase_name) {
    phase = phase_name;
    stop.store(false, std::memory_order_release);
    started.store(0, std::memory_order_relaxed);
    completed.store(0, std::memory_order_relaxed);
    lat_ms.clear();
    t0 = std::chrono::high_resolution_clock::now();
    last_snap_t = t0;
    last_snap_completed = 0;
    last_snap_started = 0;
    std::cout << "Begin OnlineSearchLoad phase=" << phase
              << " search_threads=" << NUM_SEARCH_THREADS
              << " query_num=" << query_num << std::endl;
    for (int tid = 0; tid < NUM_SEARCH_THREADS; ++tid) {
      workers.emplace_back([&, tid, query, query_num, query_dim, recall_at, L, beam_width]() {
        std::vector<TagT> tags(recall_at);
        std::vector<float> dists(recall_at);
        pipeann::QueryStats stats;
        size_t idx = (size_t) tid;
        while (!stop.load(std::memory_order_acquire)) {
          // Do not call search while remap/reload holds/waits for exclusive merge_lock;
          // otherwise continuous shared_lock admissions starve the writer.
          if (sync_index._disk_index->merge_remap_exclusive_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
          }
          size_t q = idx % query_num;
          idx += (size_t) NUM_SEARCH_THREADS;
          started.fetch_add(1, std::memory_order_relaxed);
          auto qs = std::chrono::high_resolution_clock::now();
          sync_index.search(query + q * query_dim, recall_at, 0, L, beam_width, tags.data(), dists.data(), &stats,
                            true);
          auto qe = std::chrono::high_resolution_clock::now();
          completed.fetch_add(1, std::memory_order_relaxed);
          double ms = std::chrono::duration<double, std::milli>(qe - qs).count();
          {
            std::lock_guard<std::mutex> g(lat_mu);
            if (lat_ms.size() < 200000) lat_ms.push_back(ms);
          }
        }
      });
    }
  }

  // Emit one plot-friendly latency snapshot from recent samples; does not stop load.
  void snapshot() {
    std::vector<double> copy;
    {
      std::lock_guard<std::mutex> g(lat_mu);
      copy = lat_ms;
      lat_ms.clear();
    }
    auto now = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(now - t0).count();
    double win_s = std::chrono::duration<double>(now - last_snap_t).count();
    uint64_t n_started = started.load(std::memory_order_relaxed);
    uint64_t n_completed = completed.load(std::memory_order_relaxed);
    uint64_t win_started = n_started - last_snap_started;
    uint64_t win_completed = n_completed - last_snap_completed;
    double input_qps = elapsed > 0 ? (double) n_started / elapsed : 0.0;
    double completed_qps = elapsed > 0 ? (double) n_completed / elapsed : 0.0;
    double win_input_qps = win_s > 0 ? (double) win_started / win_s : 0.0;
    double win_completed_qps = win_s > 0 ? (double) win_completed / win_s : 0.0;
    last_snap_t = now;
    last_snap_started = n_started;
    last_snap_completed = n_completed;
    int current_time = globalTimer.elapsed() / 1.0e6f - begin_time;
    std::cout << "search current time: " << current_time << std::endl;
    if (copy.empty()) {
      std::cout << "input_qps: " << input_qps << " completed_qps: " << completed_qps
                << " win_input_qps: " << win_input_qps << " win_completed_qps: " << win_completed_qps
                << " phase=" << phase << " started=" << n_started << " completed=" << n_completed
                << " elapsed_s=" << elapsed << " (no latency samples yet)" << std::endl;
      return;
    }
    std::sort(copy.begin(), copy.end());
    auto pct = [&](double p) {
      size_t i = (size_t) (p * (copy.size() - 1));
      return copy[i];
    };
    double mean = std::accumulate(copy.begin(), copy.end(), 0.0) / (double) copy.size();
    std::cout << std::setw(4) << "Ls" << std::setw(12) << "QPS " << std::setw(18) << "Mean Lat" << std::setw(12)
              << "50 Lat" << std::setw(12) << "90 Lat" << std::setw(12) << "95 Lat" << std::setw(12) << "99 Lat"
              << std::setw(12) << "99.9 Lat" << std::setw(12) << "Recall@?" << std::setw(12) << "Disk IOs"
              << std::endl;
    std::cout << "==============================================================="
                 "==============="
              << std::endl;
    // QPS column = completed throughput over the snapshot window.
    std::cout << std::setw(4) << 0 << std::setw(12) << (float) win_completed_qps << std::setw(18) << (float) mean
              << std::setw(12) << (float) pct(0.50) << std::setw(12) << (float) pct(0.90) << std::setw(12)
              << (float) pct(0.95) << std::setw(12) << (float) pct(0.99) << std::setw(12) << (float) pct(0.999)
              << std::setw(12) << 0 << std::setw(12) << 0 << std::endl;
    std::cout << "input_qps: " << input_qps << " completed_qps: " << completed_qps
              << " win_input_qps: " << win_input_qps << " win_completed_qps: " << win_completed_qps
              << " phase=" << phase << " started=" << n_started << " completed=" << n_completed
              << " elapsed_s=" << elapsed << " win_s=" << win_s
              << " lat_samples=" << copy.size() << std::endl;
  }

  double stop_and_join() {
    stop.store(true, std::memory_order_release);
    for (auto &w : workers) {
      if (w.joinable()) w.join();
    }
    workers.clear();
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    uint64_t n_started = started.load(std::memory_order_relaxed);
    uint64_t n_completed = completed.load(std::memory_order_relaxed);
    double input_qps = elapsed > 0 ? (double) n_started / elapsed : 0.0;
    double completed_qps = elapsed > 0 ? (double) n_completed / elapsed : 0.0;
    std::cout << "End OnlineSearchLoad phase=" << phase
              << " input_qps=" << input_qps
              << " completed_qps=" << completed_qps
              << " started=" << n_started
              << " completed=" << n_completed
              << " elapsed_s=" << elapsed
              << " search_threads=" << NUM_SEARCH_THREADS << std::endl;
    return input_qps;
  }
};

template<typename T, typename TagT>
void merge_kernel(pipeann::DynamicSSDIndex<T, TagT> &sync_index, std::string &save_path) {
  sync_index.final_merge(NUM_MERGE_THREADS, 20);
}

template<typename T, typename TagT>
void deletion_kernel(T *data_load, pipeann::DynamicSSDIndex<T, TagT> &sync_index, std::vector<TagT> &delete_vec,
                     size_t aligned_dim) {
  pipeann::Timer timer;
  size_t npts = delete_vec.size();
  std::vector<double> delete_latencies(npts, 0);
  std::cout << "Begin Delete" << std::endl;
#pragma omp parallel for num_threads(NUM_DELETE_THREADS)
  for (int64_t i = 0; i < (int64_t) delete_vec.size(); i++) {
    pipeann::Timer delete_timer;
    pipeann::QueryStats stats;
    sync_index.lazy_delete(delete_vec[i]);
    delete_latencies[i] = ((double) delete_timer.elapsed());
  }
  std::sort(delete_latencies.begin(), delete_latencies.end());
  std::cout << "10p deletion time : " << delete_latencies[(size_t) (0.10 * ((double) npts))] << " ms" << std::endl
            << "50p deletion time : " << delete_latencies[(size_t) (0.5 * ((double) npts))] << " ms" << std::endl
            << "90p deletion time : " << delete_latencies[(size_t) (0.90 * ((double) npts))] << " ms" << std::endl
            << "99p deletion time : " << delete_latencies[(size_t) (0.99 * ((double) npts))] << " ms" << std::endl
            << "99.9p deletion time : " << delete_latencies[(size_t) (0.999 * ((double) npts))] << " ms" << std::endl;
}

template<typename T, typename TagT>
void insertion_kernel(T *data_load, pipeann::DynamicSSDIndex<T, TagT> &sync_index, std::vector<TagT> &insert_vec,
                      size_t aligned_dim) {
  size_t npts = insert_vec.size();
  std::vector<double> insert_latencies(npts, 0);
  pipeann::InsertStats *insert_stats = new pipeann::InsertStats[npts];
  std::cout << "Begin Insert" << std::endl;
  std::atomic_size_t success(0);
  std::atomic<bool> progress_done(false);
  std::thread progress_thread([&]() {
    while (!progress_done.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::seconds(5));
      if (!progress_done.load(std::memory_order_acquire)) {
        size_t done = success.load(std::memory_order_acquire);
        std::cout << "[insert progress] " << done << " / " << npts << " ("
                  << std::fixed << std::setprecision(3) << (100.0 * done / npts) << "%)" << std::endl;
      }
    }
  });
  pipeann::Timer timer;
#pragma omp parallel for num_threads(NUM_INSERT_THREADS)
  for (int64_t i = 0; i < (int64_t) insert_vec.size() ; i++) {
    pipeann::Timer insert_timer;
    sync_index.insert(data_load + aligned_dim * i, insert_vec[i], &insert_stats[i]);
    success++;
    insert_latencies[i] = ((double) insert_timer.elapsed());
  }

  float time_secs = timer.elapsed() / 1.0e6f;
  progress_done.store(true, std::memory_order_release);
  progress_thread.join();
  std::sort(insert_latencies.begin(), insert_latencies.end());
  std::cout << "Inserted " << insert_vec.size() << " points in " << time_secs << "s" << std::endl;
  std::cout << "10p insertion time : " << insert_latencies[(size_t) (0.10 * ((double) npts))] << " us" << std::endl
            << "50p insertion time : " << insert_latencies[(size_t) (0.5 * ((double) npts))] << " us" << std::endl
            << "90p insertion time : " << insert_latencies[(size_t) (0.90 * ((double) npts))] << " us" << std::endl
            << "99p insertion time : " << insert_latencies[(size_t) (0.99 * ((double) npts))] << " us" << std::endl
            << "99.9p insertion time : " << insert_latencies[(size_t) (0.999 * ((double) npts))] << " us" << std::endl;

  // Print InsertStats averages.
  auto avg = [&](std::function<double(const pipeann::InsertStats &)> fn) {
    double sum = 0;
    for (size_t i = 0; i < npts; i++) sum += fn(insert_stats[i]);
    return sum / npts;
  };
  auto sum = [&](std::function<double(const pipeann::InsertStats &)> fn) {
    double sum = 0;
    for (size_t i = 0; i < npts; i++) sum += fn(insert_stats[i]);
    return sum ;
  };
  const double total_avg = avg([](const auto &s) { return s.total_us; });
  const double search_avg = avg([](const auto &s) { return s.search_us; });
  const double search_rerank_avg = avg([](const auto &s) { return s.search_rerank_us; });
  const double vec_avg = avg([](const auto &s) { return s.vec_write_us; });
  const double graph_commit_avg = avg([](const auto &s) { return s.update_topo_us; });
  const double other_avg = total_avg - search_avg - vec_avg - graph_commit_avg;
  std::cout << "  total_avg_us=" << (int)total_avg << " search_avg_us=" << (int)search_avg
            << " vec_avg_us=" << (int)vec_avg << " graph_commit_avg_us=" << (int)graph_commit_avg
            << " other_avg_us=" << (int)other_avg << std::endl
            << "    search_detail: traversal=" << (int)(search_avg - search_rerank_avg)
            << " rerank=" << (int)search_rerank_avg << std::endl
            << "    graph_commit: find_loc=" << (int)avg([](const auto &s) { return s.find_graph_loc_us; })
            << " extra_lock=" << (int)avg([](const auto &s) { return s.lock_wait_us; })
            << " read=" << (int)avg([](const auto &s) { return s.read_topo_us; })
            << " prune=" << (int)avg([](const auto &s) { return s.prune_us; })
            << " ensure_vec=" << (int)avg([](const auto &s) { return s.ensure_vec_us; })
            << " publish_cache=" << (int)avg([](const auto &s) { return s.publish_cache_us; })
            << " write_disk=" << (int)avg([](const auto &s) { return s.write_topo_us; }) << std::endl
            << "    find_loc detail: nbr_lock=" << (int)avg([](const auto &s) { return s.find_loc_lock_us; })
            << " alloc_call=" << (int)avg([](const auto &s) { return s.find_loc_alloc_us; })
            << " alloc_lock_wait=" << (int)avg([](const auto &s) { return s.find_loc_alloc_wait_us; })
            << " sum_retries=" << sum([](const auto &s) { return s.find_loc_retries; }) << std::endl;

  delete[] insert_stats;
}

template<typename T, typename TagT = uint32_t>
void get_trace(std::string data_bin, uint64_t l_start, uint64_t r_start, uint64_t n, std::vector<TagT> &delete_tags,
               std::vector<TagT> &insert_tags, std::vector<T> &data_load) {
  for (uint64_t i = l_start; i < l_start + n; ++i) {
    delete_tags.push_back(i);
  }

  for (uint64_t i = r_start; i < r_start + n; ++i) {
    insert_tags.push_back(i);
  }

  // load data, load n vecs from r_start.
  int npts_i32, dim_i32;
  std::ifstream reader(data_bin, std::ios::binary | std::ios::ate);
  reader.seekg(0, reader.beg);
  reader.read((char *) &npts_i32, sizeof(int));
  reader.read((char *) &dim_i32, sizeof(int));

  size_t data_dim = dim_i32;
  data_load.resize(n * data_dim);
  reader.seekg(2 * sizeof(int) + r_start * data_dim * sizeof(T), reader.beg);
  reader.read((char *) data_load.data(), sizeof(T) * n * data_dim);
}

template<typename T, typename TagT>
void update(const std::string &data_bin, const unsigned L_disk, unsigned update_size, const unsigned nodes_to_cache,
            std::string &save_path, const std::string &query_file, std::string &truthset_file, const int recall_at,
            std::vector<uint64_t> Lsearch, const unsigned beam_width, pipeann::Distance<T> *dist_cmp,
            int search_mode = BEAM_SEARCH, int num_batches = 1, int start_batch = 0) {
  pipeann::IndexBuildParameters paras;
  paras.set(0, L_disk, 384, 1.2, 128, true, beam_width);  // 128 threads for less contention.
  std::vector<T> data_load;
  size_t dim{}, aligned_dim{};

  pipeann::Timer timer;

  std::cout << "Loading queries " << std::endl;
  T *query = NULL;
  size_t query_num, query_dim;
  pipeann::load_bin<T>(query_file, query, query_num, query_dim);

  dim = query_dim;
  aligned_dim = query_dim;
  pipeann::Metric metric = pipeann::Metric::L2;
  pipeann::DynamicSSDIndex<T, TagT> sync_index(paras, save_path, save_path + "_merge", dist_cmp, metric,
                                               search_mode);

  std::cout << "Searching before inserts: " << std::endl;

  if (start_batch < 0) {
    std::cerr << "ERROR: start_batch must be >= 0" << std::endl;
    exit(-1);
  }
  uint64_t res = (uint64_t)start_batch * (uint64_t)update_size;
  std::cout << "start_batch: " << start_batch << ", initial res(offset): " << res << std::endl;

  std::string currentFileName = GetTruthFileName(truthset_file, res);
  begin_time = globalTimer.elapsed() / 1.0e6f;
  ShowMemoryStatus(sync_index._disk_index_prefix_in);

  std::vector<double> ref_diskio;
  for (uint64_t j = 0; j < Lsearch.size(); ++j) {
    double diskio = 0;
    sync_search_kernel(query, query_num, query_dim, recall_at, Lsearch[j], beam_width, sync_index, currentFileName,
                       false, true, diskio);
    ref_diskio.push_back(diskio);
  }
  // exit (0);

  int batch = num_batches;
  std::cout << "num_batches: " << batch << std::endl;
  int inMemorySize = 0;
  std::future<void> merge_future;
  uint64_t index_npts = sync_index._disk_index->meta_.npoints;
  uint64_t vecs_per_step = update_size;
  int end_batch = start_batch + batch;
  std::cout << "update_size (vecs per round): " << vecs_per_step
            << ", num_batches: " << batch
            << ", start_batch: " << start_batch
            << ", end_batch(exclusive): " << end_batch << std::endl;
  for (int i = start_batch; i < end_batch; i++) {
    std::cout << "Batch: " << i << " Total Batch : " << end_batch << std::endl;
    std::vector<unsigned> insert_vec;
    std::vector<unsigned> delete_vec;

    /**Prepare for update*/
    uint64_t st = vecs_per_step * i;
    get_trace<T, TagT>(data_bin, st, st + index_npts, vecs_per_step, delete_vec, insert_vec, data_load);

    // Online mixed: continuous search injection during the whole insert+delete window.
    OnlineSearchLoad<T, TagT> online_upd;
    online_upd.start(query, query_num, query_dim, recall_at, Lsearch[0], beam_width, sync_index, "update");

    std::future<void> insert_future = std::async(std::launch::async, insertion_kernel<T, TagT>, data_load.data(),
                                                 std::ref(sync_index), std::ref(insert_vec), aligned_dim);

    std::future<void> delete_future = std::async(std::launch::async, deletion_kernel<T, TagT>, data_load.data(),
                                                 std::ref(sync_index), std::ref(delete_vec), aligned_dim);

    std::future_status insert_status;
    std::future_status delete_status;
    do {
      insert_status = insert_future.wait_for(std::chrono::seconds(5));
      delete_status = delete_future.wait_for(std::chrono::seconds(5));
      if (insert_status == std::future_status::deferred || delete_status == std::future_status::deferred) {
        std::cout << "deferred\n";
      } else if (insert_status == std::future_status::timeout || delete_status == std::future_status::timeout) {
        ShowMemoryStatus(sync_index._disk_index_prefix_in);
        online_upd.snapshot();  // plot point + running input_qps
      }
      if (insert_status == std::future_status::ready) {
        std::cout << "Insertions complete!\n";
      }
      if (delete_status == std::future_status::ready) {
        std::cout << "Deletions complete!\n";
      }
    } while (insert_status != std::future_status::ready || delete_status != std::future_status::ready);
    double upd_input_qps = online_upd.stop_and_join();
    std::cout << "online_mixed_update_input_qps: " << upd_input_qps << std::endl;

    inMemorySize += insert_vec.size();

    std::cout << "Search after update, current vector number: " << res << std::endl;
    
    sync_index._disk_index->verify_id2loc();

    // Dump snapshot of newly inserted points for comparison.
    // {
    //   std::string snap_path = save_path + "_snapshot_" + std::to_string(i) + ".bin";
    //   sync_index.debug_snapshot(snap_path);
    // }

    res += vecs_per_step;
    currentFileName = GetTruthFileName(truthset_file, res);

    std::vector<double> disk_ios;
    for (uint64_t j = 0; j < Lsearch.size(); ++j) {
      double diskio = 0;
      sync_search_kernel(query, query_num, query_dim, recall_at, Lsearch[j], beam_width, sync_index, currentFileName,
                         false, true, diskio);
      disk_ios.push_back(diskio);
    }
    // exit (-1);

    bool last_batch = (i == end_batch - 1);
    bool should_merge =
        last_batch ||
        (i % MERGE_ROUND == MERGE_ROUND - 1 ||
         disk_ios[0] / ref_diskio[0] > MERGE_IO_THRESHOLD);
    if (should_merge) {
      std::cout << "Begin Merge" << std::endl;
      // Online search through merge; DGAI uses page_lock_table, Odin brief remap exclusive.
      std::cout << "merge_policy: online_search_page_locks_or_brief_remap" << std::endl;
      OnlineSearchLoad<T, TagT> online_merge;
      online_merge.start(query, query_num, query_dim, recall_at, Lsearch[0], beam_width, sync_index, "merge");
      merge_future = std::async(std::launch::async, merge_kernel<T, TagT>, std::ref(sync_index), std::ref(save_path));
      std::this_thread::sleep_for(std::chrono::seconds(5));
      std::cout << "Sending Merge" << std::endl;
      inMemorySize = 0;
      std::future_status merge_status;
      do {
        merge_status = merge_future.wait_for(std::chrono::seconds(10));
        ShowMemoryStatus(sync_index._disk_index_prefix_in);
        online_merge.snapshot();
      } while (merge_status != std::future_status::ready);
      // Merge finished (reload unlocked merge_lock). Stop load before post-merge batch search.
      double merge_input_qps = online_merge.stop_and_join();
      std::cout << "online_mixed_merge_input_qps: " << merge_input_qps << std::endl;
      merge_future.get();

      ref_diskio.clear();
      for (uint32_t j = 0; j < Lsearch.size(); ++j) {
        double diskio;
        sync_search_kernel(query, query_num, query_dim, recall_at, Lsearch[j], beam_width, sync_index, currentFileName,
                           false, true, diskio);
        ref_diskio.push_back(diskio);
      }
      std::cout << "Merge finished " << std::endl;
#ifdef DGAI_STRICT_GRAPH_CHECKS
      sync_index._disk_index->verify_graph_neighbor_ids();
      sync_index._disk_index->log_graph_stats();
#endif
    }

    if (last_batch) {
      std::cout << "Done" << std::endl;
      exit(0);
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 12) {
    std::cout << "Correct usage: " << argv[0] << " <type[int8/uint8/float]> <data_bin> <L_disk> "
              << " <indice_path> <query_file> <truthset_prefix> <recall@>"
              << " <#beam_width> <update_size> <num_batches> <Lsearch> [search_mode=0] [start_batch=0]"
              << "\n  update_size: delete+insert count per round (batch_npts * UPDATE_RATIO)"
              << "\n  num_batches: number of update rounds"
              << "\n  search_mode: 0=beam(default), 4=decouple, 5=decouple_beam, 7=starling_decouple" << std::endl;
    exit(-1);
  }

  int arg_no = 2;
  std::string data_bin = std::string(argv[arg_no++]);
  unsigned L_disk = (unsigned) atoi(argv[arg_no++]);
  std::string save_path(argv[arg_no++]);

  std::string query_file(argv[arg_no++]);
  std::string truthset(argv[arg_no++]);
  int recall_at = (int) std::atoi(argv[arg_no++]);
  unsigned beam_width = (unsigned) std::atoi(argv[arg_no++]);
  unsigned update_size = (unsigned) std::atoi(argv[arg_no++]);
  int num_batches = (int) std::atoi(argv[arg_no++]);
  if (update_size == 0) {
    std::cerr << "ERROR: update_size must be > 0 (batch_npts * UPDATE_RATIO)" << std::endl;
    exit(-1);
  }
  if (num_batches <= 0) {
    std::cerr << "ERROR: num_batches must be > 0" << std::endl;
    exit(-1);
  }

  // Optional search_mode argument (default: BEAM_SEARCH=0).
  // Modes: 0=beam, 1=page, 2=pipe, 3=coro, 4=decouple, 5=decouple_beam,
  //        6=decouple_page, 7=starling_decouple, 8=non_decouple.
  int search_mode = BEAM_SEARCH;
  if (arg_no < argc) {
    // Distinguish search_mode (0..8) from Lsearch values (typically >= 10).
    int val = std::atoi(argv[arg_no]);
    if (val >= 0 && val <= NON_DECOUPLE_SEARCH) {
      search_mode = val;
      arg_no++;
    }
  }

  std::vector<uint64_t> Lsearch;
  for (int i = arg_no; i < argc; ++i) {
    Lsearch.push_back(std::atoi(argv[i]));
  }

  unsigned nodes_to_cache = 0;

  // Optional resume offset via env (keeps CLI compatible with existing scripts).
  int start_batch = 0;
  if (const char *sb = std::getenv("DGAI_START_BATCH")) {
    start_batch = std::atoi(sb);
  }
  std::cout << "DGAI_START_BATCH=" << start_batch << std::endl;

  if (std::string(argv[1]) == std::string("int8")) {
    pipeann::DistanceL2Int8 dist_cmp;
    update<int8_t, unsigned>(data_bin, L_disk, update_size, nodes_to_cache, save_path, query_file, truthset, recall_at,
                             Lsearch, beam_width, &dist_cmp, search_mode, num_batches, start_batch);
  } else if (std::string(argv[1]) == std::string("uint8")) {
    pipeann::DistanceL2UInt8 dist_cmp;
    update<uint8_t, unsigned>(data_bin, L_disk, update_size, nodes_to_cache, save_path, query_file, truthset, recall_at,
                              Lsearch, beam_width, &dist_cmp, search_mode, num_batches, start_batch);
  } else if (std::string(argv[1]) == std::string("float")) {
    pipeann::DistanceL2Float dist_cmp;
    update<float, unsigned>(data_bin, L_disk, update_size, nodes_to_cache, save_path, query_file, truthset, recall_at, Lsearch,
                            beam_width, &dist_cmp, search_mode, num_batches, start_batch);
  } else
    std::cout << "Unsupported type. Use float/int8/uint8" << std::endl;
}
