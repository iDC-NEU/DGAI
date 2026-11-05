#include "ssd_index.h"
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

int NUM_INSERT_THREADS = 10;
int NUM_SEARCH_THREADS = 32;

int search_mode = BEAM_SEARCH;

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

  LOG(INFO) << " memory current time: " << current_time << " RSS : " << rss << " KB " << index_file_name
            << " Index size " << (st.st_size / (1 << 20)) << " MB";
}

std::string convertFloatToString(const float value, const int precision = 0) {
  std::stringstream stream{};
  stream << std::fixed << std::setprecision(precision) << value;
  return stream.str();
}

std::string GetTruthFileName(const std::string &truthFilePrefix, int l_start) {
  std::string fileName(truthFilePrefix);
  fileName = fileName + "/gt_" + std::to_string(l_start) + ".bin";
  LOG(INFO) << "Truth file name: " << fileName;
  return fileName;
}

template<typename T, typename TagT>
void sync_search_kernel(T *query, size_t query_num, size_t query_dim, const int recall_at, uint32_t mem_L, uint64_t L,
                        uint32_t beam_width, pipeann::DynamicSSDIndex<T, TagT> &sync_index, std::string &truthset_file,
                        bool merged, bool calRecall, double &disk_io) {
  if (NUM_SEARCH_THREADS == 0) {
    return;
  }
  unsigned *gt_ids = NULL;
  float *gt_dists = NULL;
  size_t gt_num, gt_dim;

  if (!file_exists(truthset_file)) {
    calRecall = false;
  }

  if (calRecall) {
    LOG(INFO) << "current truthfile: " << truthset_file;
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
  std::cerr << std::setw(4) << "Ls" << std::setw(12) << "QPS " << std::setw(18) << "Mean Lat" << std::setw(12)
            << "50 Lat" << std::setw(12) << "90 Lat" << std::setw(12) << "95 Lat" << std::setw(12) << "99 Lat"
            << std::setw(12) << "99.9 Lat" << std::setw(12) << recall_string << std::setw(12) << "Disk IOs"
            << std::endl;
  std::cerr << "==============================================================="
               "==============="
            << std::endl;
  auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(NUM_SEARCH_THREADS) schedule(dynamic)
  for (int64_t i = 0; i < (int64_t) query_num; i++) {
    sync_index.search(query + i * query_dim, recall_at, mem_L, L, beam_width, query_result_tags + i * recall_at,
                      query_result_dists + i * recall_at, stats + i, true);

    latency_stats[i] = stats[i].total_us / 1000.0;  // convert to ms
    if (search_mode == BEAM_SEARCH) {
      // Here we follow the original paper 's settings...
      // For PipeSearch, do not sleep is faster.
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
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

  float mean_ios =
      (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.n_ios; });

  std::sort(latency_stats.begin(), latency_stats.end());
  std::cerr << std::setw(4) << L << std::setw(12) << qps << std::setw(18)
            << ((float) std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0f)) / (float) query_num
            << std::setw(12) << (float) latency_stats[(uint64_t) (0.50 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(uint64_t) (0.90 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(uint64_t) (0.95 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(uint64_t) (0.99 * ((double) query_num))] << std::setw(12)
            << (float) latency_stats[(uint64_t) (0.999 * ((double) query_num))] << std::setw(12) << recall
            << std::setw(12) << mean_ios << std::endl;

  LOG(INFO) << "search current time: " << current_time;
  disk_io = mean_ios;

  delete[] query_result_dists;
  delete[] query_result_tags;
  delete[] stats;
}

template<typename T, typename TagT>
void merge_kernel(pipeann::DynamicSSDIndex<T, TagT> &sync_index) {
  sync_index.final_merge(NUM_INSERT_THREADS);
}

template<typename T, typename TagT>
void insertion_kernel(T *data_load, pipeann::DynamicSSDIndex<T, TagT> &sync_index, std::vector<TagT> &insert_vec,
                      size_t dim) {
  pipeann::Timer timer;
  size_t npts = insert_vec.size();
  std::vector<double> insert_latencies(npts, 0);
  LOG(INFO) << "Begin Insert";
  std::atomic_size_t success(0);

#pragma omp parallel for num_threads(NUM_INSERT_THREADS)
  for (int64_t i = 0; i < (int64_t) insert_vec.size(); i++) {
    pipeann::Timer insert_timer;
    sync_index.insert(data_load + dim * i, insert_vec[i]);
    success++;
    insert_latencies[i] = ((double) insert_timer.elapsed());
  }

  float time_secs = timer.elapsed() / 1.0e6f;
  std::sort(insert_latencies.begin(), insert_latencies.end());
  LOG(INFO) << "Inserted " << insert_vec.size() << " points in " << time_secs << "s";
  LOG(INFO) << "10p insertion time : " << insert_latencies[(size_t) (0.10 * ((double) npts))] << " us";
  LOG(INFO) << "50p insertion time : " << insert_latencies[(size_t) (0.5 * ((double) npts))] << " us";
  LOG(INFO) << "90p insertion time : " << insert_latencies[(size_t) (0.90 * ((double) npts))] << " us";
  LOG(INFO) << "99p insertion time : " << insert_latencies[(size_t) (0.99 * ((double) npts))] << " us";
  LOG(INFO) << "99.9p insertion time : " << insert_latencies[(size_t) (0.999 * ((double) npts))] << " us";
}

template<typename T, typename TagT = uint32_t>
void get_trace(std::string data_bin, uint64_t l_start, uint64_t r_start, uint64_t n, std::vector<TagT> &delete_tags,
               std::vector<TagT> &insert_tags, std::vector<T> &data_load) {
  LOG(INFO) << "l_start: " << l_start << " r_start: " << r_start << " n: " << n;

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
void update(const std::string &data_bin, const unsigned L_disk, int vecs_per_step, int num_steps,
            const std::string &index_prefix, const std::string &query_file, const std::string &truthset_file,
            size_t truthset_l_offset, const int recall_at, const std::vector<uint64_t> &Lsearch,
            const unsigned beam_width, const uint32_t search_beam_width, const uint32_t search_mem_L,
            pipeann::Distance<T> *dist_cmp) {
  pipeann::IndexBuildParameters paras;
  paras.set(0, L_disk, 384, 1.2, NUM_SEARCH_THREADS + NUM_INSERT_THREADS, true, beam_width);
  std::vector<T> data_load;
  size_t dim{};

  pipeann::Timer timer;

  LOG(INFO) << "Loading queries";
  T *query = NULL;
  size_t query_num, query_dim;
  pipeann::load_bin<T>(query_file, query, query_num, query_dim);

  dim = query_dim;
  pipeann::Metric metric = pipeann::Metric::L2;
  pipeann::DynamicSSDIndex<T, TagT> sync_index(paras, index_prefix, index_prefix + "_merge", dist_cmp, metric,
                                               search_mode, (search_mem_L > 0));

  LOG(INFO) << "Searching before inserts: ";

  uint64_t res = 0;

  std::string currentFileName = GetTruthFileName(truthset_file, res + truthset_l_offset);
  begin_time = globalTimer.elapsed() / 1.0e6f;
  ShowMemoryStatus(sync_index._disk_index_prefix_in);

  std::vector<double> ref_diskio;
  for (size_t j = 0; j < Lsearch.size(); ++j) {
    double diskio = 0;
    sync_search_kernel(query, query_num, query_dim, recall_at, search_mem_L, Lsearch[j], search_beam_width, sync_index,
                       currentFileName, false, true, diskio);
    ref_diskio.push_back(diskio);
  }

  int inMemorySize = 0;
  std::future<void> merge_future;
  uint64_t index_npts = sync_index._disk_index->meta_.npoints;
  for (int i = 0; i < num_steps; i++) {
    LOG(INFO) << "Batch: " << i << " Total Batch : " << num_steps;
    std::vector<unsigned> insert_vec;
    std::vector<unsigned> delete_vec;

    /**Prepare for update*/
    uint64_t st = vecs_per_step * i;
    uint64_t ed = st + index_npts;
    LOG(INFO) << "st: " << st << " ed: " << ed;
    get_trace<T, TagT>(data_bin, st, ed, vecs_per_step, delete_vec, insert_vec, data_load);

    std::future<void> insert_future = std::async(std::launch::async, insertion_kernel<T, TagT>, data_load.data(),
                                                 std::ref(sync_index), std::ref(insert_vec), dim);

    int total_queries = 0;
    std::future_status insert_status;
    do {
      insert_status = insert_future.wait_for(std::chrono::seconds(5));
      if (insert_status == std::future_status::deferred) {
        LOG(INFO) << "deferred\n";
      } else if (insert_status == std::future_status::timeout) {
        ShowMemoryStatus(sync_index._disk_index_prefix_in);
        LOG(INFO) << "Number of vectors: " << sync_index._disk_index->cur_id;
        double dummy;
        // for (uint32_t j = 0; j < Lsearch.size(); ++j) {
        sync_search_kernel(query, query_num, query_dim, recall_at, search_mem_L, Lsearch[0], search_beam_width,
                           sync_index, currentFileName, false, false, dummy);
        sleep(1);
        // }
        total_queries += query_num;
        LOG(INFO) << "Queries processed: " << total_queries;
      }
      if (insert_status == std::future_status::ready) {
        LOG(INFO) << "Insertions complete!\n";
      }
    } while (insert_status != std::future_status::ready);

    inMemorySize += insert_vec.size();

    LOG(INFO) << "Search after update, current vector number: " << res;

    res += vecs_per_step;
    currentFileName = GetTruthFileName(truthset_file, res + truthset_l_offset);

    std::vector<double> disk_ios;
    for (size_t j = 0; j < Lsearch.size(); ++j) {
      double diskio = 0;
      sync_search_kernel(query, query_num, query_dim, recall_at, search_mem_L, Lsearch[j], search_beam_width,
                         sync_index, currentFileName, false, true, diskio);
      disk_ios.push_back(diskio);
    }

    if (i == num_steps - 1 && num_steps >= 10) {  // store the last index, figs 9 and 11 uses num_steps < 10.
      LOG(INFO) << "Store the last index to disk.";
      merge_future = std::async(std::launch::async, merge_kernel<T, TagT>, std::ref(sync_index));
      std::future_status merge_status;
      do {
        merge_status = merge_future.wait_for(std::chrono::seconds(10));
      } while (merge_status != std::future_status::ready);
      LOG(INFO) << "Store finished.";
      exit(0);
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 18) {
    LOG(INFO) << "Correct usage: " << argv[0] << " <type[int8/uint8/float]> <data_bin> <L_disk>"
              << " <vecs_per_step> <num_steps> <insert_threads> <search_threads> <search_mode>"
              << " <index_prefix> <query_file> <truthset_prefix> <truthset_l_offset> <recall@>"
              << " <#beam_width> <search_beam_width> <mem_L> <Lsearch> <L2>";
    exit(-1);
  }

  int arg_no = 2;
  std::string data_bin = std::string(argv[arg_no++]);
  unsigned L_disk = (unsigned) atoi(argv[arg_no++]);

  // 1M vectors per step.
  int vecs_per_step = (int) std::atoi(argv[arg_no++]);

  // 100 steps for 100M + 100M test, 200 steps for 800M + 200M test.
  int num_steps = (int) std::atoi(argv[arg_no++]);

  NUM_INSERT_THREADS = (int) std::atoi(argv[arg_no++]);
  NUM_SEARCH_THREADS = (int) std::atoi(argv[arg_no++]);
  LOG(INFO) << "num insert threads: " << NUM_INSERT_THREADS;
  LOG(INFO) << "num search threads: " << NUM_SEARCH_THREADS;

  search_mode = std::atoi(argv[arg_no++]);
  LOG(INFO) << "search mode: " << search_mode;

  std::string index_prefix(argv[arg_no++]);

  std::string query_file(argv[arg_no++]);

  // truthsets lie in the same folder, with gt_X.bin as the truthset of [0, x + 100M) vectors.
  std::string truthset(argv[arg_no++]);

  // We generate truth files every 1M vectors, gt_X.bin is the truthset of [0, x + 100M) vectors.
  // This should be 0 for 100M + 100M test, and 700M for 800M + 200M test.
  size_t truthset_l_offset = (size_t) std::atoll(argv[arg_no++]);
  int recall_at = (int) std::atoi(argv[arg_no++]);
  unsigned beam_width = (unsigned) std::atoi(argv[arg_no++]);
  unsigned search_beam_width = (unsigned) std::atoi(argv[arg_no++]);
  unsigned search_mem_L = (unsigned) std::atoi(argv[arg_no++]);  // 0 if no mem_index is used.
  std::vector<uint64_t> Lsearch;
  for (int i = arg_no; i < argc; ++i) {
    Lsearch.push_back(std::atoi(argv[i]));
  }

  unsigned nodes_to_cache = 0;

  if (std::string(argv[1]) == std::string("int8")) {
    pipeann::DistanceL2Int8 dist_cmp;
    update<int8_t, unsigned>(data_bin, L_disk, vecs_per_step, num_steps, index_prefix, query_file, truthset,
                             truthset_l_offset, recall_at, Lsearch, beam_width, search_beam_width, search_mem_L,
                             &dist_cmp);
  } else if (std::string(argv[1]) == std::string("uint8")) {
    pipeann::DistanceL2UInt8 dist_cmp;
    update<uint8_t, unsigned>(data_bin, L_disk, vecs_per_step, num_steps, index_prefix, query_file, truthset,
                              truthset_l_offset, recall_at, Lsearch, beam_width, search_beam_width, search_mem_L,
                              &dist_cmp);
  } else if (std::string(argv[1]) == std::string("float")) {
    pipeann::DistanceL2Float dist_cmp;
    update<float, unsigned>(data_bin, L_disk, vecs_per_step, num_steps, index_prefix, query_file, truthset,
                            truthset_l_offset, recall_at, Lsearch, beam_width, search_beam_width, search_mem_L,
                            &dist_cmp);
  } else
    LOG(INFO) << "Unsupported type. Use float/int8/uint8";
}
