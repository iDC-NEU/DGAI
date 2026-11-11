#include <cstring>
#include <omp.h>
#include <ssd_index.h>
#include <iomanip>

#include "utils/log.h"
#include "filter/selector.h"
#include "filter/label.h"
#include "nbr/nbr.h"
#include "utils/timer.h"
#include "utils.h"
#include "linux_aligned_file_reader.h"

template<typename T>
int search_disk_index(int argc, char **argv) {
  T *query = nullptr;
  unsigned *gt_ids = nullptr;
  float *gt_dists = nullptr;
  uint32_t *tags = nullptr;
  size_t query_num, query_dim, gt_num, gt_dim;
  std::vector<uint64_t> Lvec;

  int index = 2;
  std::string index_prefix_path(argv[index++]);
  uint32_t num_threads = std::atoi(argv[index++]);
  uint32_t beamwidth = std::atoi(argv[index++]);
  std::string query_bin(argv[index++]);
  std::string truthset_bin(argv[index++]);
  uint64_t recall_at = std::atoi(argv[index++]);
  std::string dist_metric(argv[index++]);
  std::string nbr_type = argv[index++];
  std::string selector_type = argv[index++];
  std::string query_label_file = argv[index++];
  uint64_t relaxed_monotonicity_lmax = std::atoi(argv[index++]);
  uint32_t mem_L = std::atoi(argv[index++]);

  pipeann::Metric m = pipeann::get_metric(dist_metric);

  for (int ctr = index; ctr < argc; ctr++) {
    uint64_t curL = std::atoi(argv[ctr]);
    if (relaxed_monotonicity_lmax >= recall_at || curL >= recall_at)
      Lvec.push_back(curL);
  }

  if (Lvec.empty()) {
    LOG(ERROR) << "No valid Lsearch found. Lsearch must be >= recall_at";
    return -1;
  }

  LOG(INFO) << "Search parameters: threads=" << num_threads << ", beamwidth=" << beamwidth;

  // Load query vectors
  pipeann::load_bin<T>(query_bin, query, query_num, query_dim);

  // Load ground truth if available
  bool calc_recall_flag = false;
  if (file_exists(truthset_bin)) {
    pipeann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim, &tags);
    if (gt_num != query_num) {
      LOG(ERROR) << "Mismatch in number of queries and ground truth data";
    }
    calc_recall_flag = true;
  }

  // Load query labels from spmat file
  pipeann::SpmatLabel query_labels(query_label_file);
  LOG(INFO) << "Loaded query labels: " << query_labels.labels_.size() << " queries";

  // Get selector based on selector_type
  pipeann::AbstractSelector *selector = pipeann::get_selector<T>(selector_type);
  if (selector == nullptr) {
    LOG(ERROR) << "Unknown selector type: " << selector_type;
    return -1;
  }
  LOG(INFO) << "Using selector: " << selector_type;

  // Initialize index
  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  pipeann::AbstractNeighbor<T> *nbr_handler = pipeann::get_nbr_handler<T>(m, nbr_type);
  std::unique_ptr<pipeann::SSDIndex<T>> _pFlashIndex(new pipeann::SSDIndex<T>(m, reader, nbr_handler, true));

  if (_pFlashIndex->load(index_prefix_path.c_str(), num_threads, false) != 0) {
    return -1;
  }

  if (mem_L != 0) {
    auto mem_index_path = index_prefix_path + "_mem.index";
    LOG(INFO) << "Load memory index from " << mem_index_path;
    _pFlashIndex->load_mem_index(mem_index_path);
  }

  omp_set_num_threads(num_threads);

  // Prepare filter data buffers (one per query)
  size_t max_filter_size = query_labels.label_size();
  std::vector<std::vector<char>> filter_buffers(query_num);
  for (size_t i = 0; i < query_num; i++) {
    filter_buffers[i].resize(max_filter_size, 0);
    query_labels.write(i, filter_buffers[i].data());
  }

  std::vector<std::vector<uint32_t>> query_result_tags(Lvec.size());
  std::vector<std::vector<float>> query_result_dists(Lvec.size());

  auto run_tests = [&](uint32_t test_id, bool output) {
    pipeann::QueryStats *stats = new pipeann::QueryStats[query_num];
    uint64_t L = Lvec[test_id];

    query_result_tags[test_id].resize(recall_at * query_num);
    query_result_dists[test_id].resize(recall_at * query_num);

    auto s = std::chrono::high_resolution_clock::now();

    if (relaxed_monotonicity_lmax == 0) {
#pragma omp parallel for schedule(dynamic, 1)
      for (int64_t i = 0; i < (int64_t) query_num; i++) {
        _pFlashIndex->pipe_search(query + (i * query_dim), (uint64_t) recall_at, mem_L, (uint64_t) L,
                                  query_result_tags[test_id].data() + (i * recall_at),
                                  query_result_dists[test_id].data() + (i * recall_at), (uint64_t) beamwidth, stats + i,
                                  selector, filter_buffers[i].data(), 0);
      }
    } else {
      // Here, we use relaxed_monotonicity_lmax as the maximum candidate pool length.
      // When there are L search results after converge, the search terminates.
#pragma omp parallel for schedule(dynamic, 1)
      for (int64_t i = 0; i < (int64_t) query_num; i++) {
        _pFlashIndex->pipe_search(query + (i * query_dim), (uint64_t) recall_at, mem_L, relaxed_monotonicity_lmax,
                                  query_result_tags[test_id].data() + (i * recall_at),
                                  query_result_dists[test_id].data() + (i * recall_at), (uint64_t) beamwidth, stats + i,
                                  selector, filter_buffers[i].data(), (uint64_t) L);
      }
    }

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    float qps = (float) query_num / diff.count();

    float mean_latency =
        pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &s) { return s.total_us; });
    float latency_999 = pipeann::get_percentile_stats(stats, query_num, 0.999f,
                                                      [](const pipeann::QueryStats &s) { return s.total_us; });
    float mean_hops = pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &s) { return s.n_hops; });
    float mean_ios = pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &s) { return s.n_ios; });

    delete[] stats;

    if (output) {
      float recall = 0;
      if (calc_recall_flag) {
        recall =
            pipeann::calculate_recall((uint32_t) query_num, gt_ids, gt_dists, (uint32_t) gt_dim,
                                      query_result_tags[test_id].data(), (uint32_t) recall_at, (uint32_t) recall_at);
      }

      std::cout << std::setw(6) << L << std::setw(12) << beamwidth << std::setw(12) << qps << std::setw(12)
                << mean_latency << std::setw(12) << latency_999 << std::setw(12) << mean_hops << std::setw(12)
                << mean_ios;
      if (calc_recall_flag) {
        std::cout << std::setw(12) << recall;
      }
      std::cout << std::endl;
    }
  };

  // Print header
  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(6) << "L" << std::setw(12) << "I/O Width" << std::setw(12) << "QPS" << std::setw(12)
            << "AvgLat(us)" << std::setw(12) << "P99 Lat" << std::setw(12) << "Mean Hops" << std::setw(12)
            << "Mean IOs";
  if (calc_recall_flag) {
    std::cout << std::setw(12) << recall_string;
  }
  std::cout << std::endl;
  std::cout << std::string(90, '=') << std::endl;

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    run_tests(test_id, true);
  }

  delete selector;
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 14) {
    std::cout << "Usage: " << argv[0] << " <index_type (float/int8/uint8)>"
              << " <index_prefix_path>"
              << " <num_threads>"
              << " <beamwidth>"
              << " <query_file.bin>"
              << " <truthset.bin (use \"null\" for none)>"
              << " <K>"
              << " <similarity (cosine/l2/mips)>"
              << " <nbr_type (pq/rabitq)>"
              << " <selector_type (range/intersect/subset)>"
              << " <query_label.spmat>"
              << " <relaxed_monotonicity_lmax (0 means disabled)>"
              << " <mem_L (0 means no mem index)>"
              << " <L1> [L2] ..." << std::endl;
    return -1;
  }

  std::string index_type = argv[1];
  if (index_type == "float")
    return search_disk_index<float>(argc, argv);
  else if (index_type == "int8")
    return search_disk_index<int8_t>(argc, argv);
  else if (index_type == "uint8")
    return search_disk_index<uint8_t>(argc, argv);
  else {
    std::cout << "Unsupported index type: " << index_type << ". Use float/int8/uint8" << std::endl;
    return -1;
  }
}
