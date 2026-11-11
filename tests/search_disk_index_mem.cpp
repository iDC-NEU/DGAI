#include <atomic>
#include <cstring>
#include <iomanip>
#include <omp.h>
#include <ssd_index.h>
#include <set>
#include <string.h>
#include <time.h>

#include "distance.h"
#include "nbr/dummy_nbr.h"
#include "utils/log.h"
#include "index.h"
#include "utils/kmeans_utils.h"
#include "utils/partition.h"
#include "utils/timer.h"
#include "utils.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"

template<typename T>
int search_disk_index(int argc, char **argv) {
  // load query bin
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
  int search_mode = std::atoi(argv[index++]);
  bool use_page_search = search_mode == SearchMode::PAGE_SEARCH;
  uint32_t mem_L = std::atoi(argv[index++]);

  pipeann::Metric m = pipeann::get_metric(dist_metric);

  std::string disk_index_tag_file = index_prefix_path + "_disk.index.tags";

  bool calc_recall_flag = false;

  for (int ctr = index; ctr < argc; ctr++) {
    uint64_t curL = std::atoi(argv[ctr]);
    if (curL >= recall_at)
      Lvec.push_back(curL);
  }

  if (Lvec.size() == 0) {
    std::cout << "No valid Lsearch found. Lsearch must be at least recall_at" << std::endl;
    return -1;
  }

  std::cout << "Search parameters: #threads: " << num_threads << ", ";
  if (beamwidth <= 0)
    std::cout << "beamwidth to be optimized for each L value" << std::endl;
  else
    std::cout << " beamwidth: " << beamwidth << std::endl;

  pipeann::load_bin<T>(query_bin, query, query_num, query_dim);

  if (file_exists(truthset_bin)) {
    pipeann::load_truthset(truthset_bin, gt_ids, gt_dists, gt_num, gt_dim, &tags);
    if (gt_num != query_num) {
      std::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
    }
    calc_recall_flag = true;
  }

  std::shared_ptr<AlignedFileReader> reader = nullptr;
  reader.reset(new LinuxAlignedFileReader());

  pipeann::SSDIndex<T> disk_index(m, reader, new pipeann::DummyNeighbor<T>(m), true, nullptr);
  auto &_pFlashIndex = *disk_index.load_to_mem(index_prefix_path);

  LOG(INFO) << "Num threads: " << num_threads;
  omp_set_num_threads(num_threads);

  LOG(INFO) << "Use page search: " << use_page_search;

  std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
  std::cout.precision(2);

  std::string recall_string = "Recall@" + std::to_string(recall_at);
  std::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth" << std::setw(12) << "QPS" << std::setw(12)
            << "Mean Lat" << std::setw(12) << "P99 Lat" << std::setw(12) << "Mean Hops" << std::setw(12) << "Mean IOs"
            << std::setw(12) << "CPU (us)" << std::setw(12) << "CPU1 (us)" << std::setw(12) << "CPU2 (us)"
            << std::setw(12) << "IO (us)" << std::setw(12) << "IO1 (us)";
  if (calc_recall_flag) {
    std::cout << std::setw(12) << recall_string << std::endl;
  } else
    std::cout << std::endl;
  std::cout << "==============================================================="
               "==========================================="
            << std::endl;

  std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
  std::vector<std::vector<uint32_t>> query_result_tags(Lvec.size());
  std::vector<std::vector<float>> query_result_dists(Lvec.size());

  for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++) {
    uint64_t L = Lvec[test_id];

    query_result_ids[test_id].resize(recall_at * query_num);
    query_result_dists[test_id].resize(recall_at * query_num);
    query_result_tags[test_id].resize(recall_at * query_num);

    pipeann::QueryStats *stats = new pipeann::QueryStats[query_num];

    std::vector<uint64_t> query_result_tags_64(recall_at * query_num);
    std::vector<uint32_t> query_result_tags_32(recall_at * query_num);
    auto s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < (int64_t) query_num; i++) {
      auto s1 = std::chrono::high_resolution_clock::now();
      _pFlashIndex.search(query + (i * query_dim), recall_at, L, query_result_tags_32.data() + (i * recall_at),
                          query_result_dists[test_id].data() + (i * recall_at), &stats[i]);
      auto e1 = std::chrono::high_resolution_clock::now();
      stats[i].total_us = std::chrono::duration_cast<std::chrono::microseconds>(e1 - s1).count();
    }

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    float qps = (float) ((1.0 * (double) query_num) / (1.0 * (double) diff.count()));

    pipeann::convert_types<uint32_t, uint32_t>(query_result_tags_32.data(), query_result_tags[test_id].data(),
                                               (size_t) query_num, (size_t) recall_at);

    float mean_latency = (float) pipeann::get_mean_stats(
        stats, query_num, [](const pipeann::QueryStats &stats) { return stats.total_us; });

    float latency_999 = (float) pipeann::get_percentile_stats(
        stats, query_num, 0.999f, [](const pipeann::QueryStats &stats) { return stats.total_us; });

    float mean_hops = (float) pipeann::get_mean_stats(stats, query_num,
                                                      [](const pipeann::QueryStats &stats) { return stats.n_hops; });

    float mean_ios =
        (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.n_ios; });

    float mean_cpuus = (float) pipeann::get_mean_stats(stats, query_num,
                                                       [](const pipeann::QueryStats &stats) { return stats.cpu_us; });

    float mean_cpu1us = (float) pipeann::get_mean_stats(stats, query_num,
                                                        [](const pipeann::QueryStats &stats) { return stats.cpu_us1; });

    float mean_cpu2us = (float) pipeann::get_mean_stats(stats, query_num,
                                                        [](const pipeann::QueryStats &stats) { return stats.cpu_us2; });

    float mean_ious =
        (float) pipeann::get_mean_stats(stats, query_num, [](const pipeann::QueryStats &stats) { return stats.io_us; });

    float mean_io1us = (float) pipeann::get_mean_stats(stats, query_num,
                                                       [](const pipeann::QueryStats &stats) { return stats.io_us1; });
    delete[] stats;

    float recall = 0;
    if (calc_recall_flag) {
      /* Attention: in SPACEV, there may be multiple vectors with the same distance,
         which may cause lower than expected recall@1 (?) */
      recall = (float) pipeann::calculate_recall((uint32_t) query_num, gt_ids, gt_dists, (uint32_t) gt_dim,
                                                 query_result_tags[test_id].data(), (uint32_t) recall_at,
                                                 (uint32_t) recall_at);
    }

    std::cout << std::setw(6) << L << std::setw(12) << 1 << std::setw(12) << qps << std::setw(12) << mean_latency
              << std::setw(12) << latency_999 << std::setw(12) << mean_hops << std::setw(12) << mean_ios
              << std::setw(12) << mean_cpuus << std::setw(12) << mean_cpu1us << std::setw(12) << mean_cpu2us
              << std::setw(12) << mean_ious << std::setw(12) << mean_io1us;
    if (calc_recall_flag) {
      std::cout << std::setw(12) << recall << std::endl;
    }
  }
  // std::this_thread::sleep_for(std::chrono::seconds(10));

  // std::cout << "Done searching. Now saving results " << std::endl;
  // uint64_t test_id = 0;
  // for (auto L : Lvec) {
  //   std::string cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
  //   pipeann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);

  //   cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_tags_uint32.bin";
  //   pipeann::save_bin<uint32_t>(cur_result_path, query_result_tags[test_id].data(), query_num, recall_at);
  //   cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_dists_float.bin";
  //   pipeann::save_bin<float>(cur_result_path, query_result_dists[test_id++].data(), query_num, recall_at);
  // }

  // pipeann::aligned_free(query);
  // if (warmup != nullptr)
  //   pipeann::aligned_free(warmup);
  // delete[] gt_ids;
  // delete[] gt_dists;
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 12) {
    // tags == 1!
    std::cout << "Usage: " << argv[0]
              << " <index_type (float/int8/uint8)>  <index_prefix_path>"
                 " <num_threads>  <pipeline width> "
                 " <query_file.bin>  <truthset.bin (use \"null\" for none)> "
                 " <K> <similarity (cosine/l2/mips)> <nbr_type (pq/rabitq)>"
                 " <search_mode(0 for beam search / 1 for page search / 2 for pipe search)> <mem_L (0 means not "
                 "using mem index)> <L1> [L2] etc."
              << std::endl;
    exit(-1);
  }

  if (std::string(argv[1]) == std::string("float"))
    search_disk_index<float>(argc, argv);
  else if (std::string(argv[1]) == std::string("int8"))
    search_disk_index<int8_t>(argc, argv);
  else if (std::string(argv[1]) == std::string("uint8"))
    search_disk_index<uint8_t>(argc, argv);
  else
    std::cout << "Unsupported index type. Use float or int8 or uint8" << std::endl;
}
