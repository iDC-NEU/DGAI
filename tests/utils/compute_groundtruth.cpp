#include <string>
#include <iostream>
#include <fstream>
#include <cassert>

#include <vector>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <limits>
#include <queue>
#include <stdlib.h>

#include "omp.h"
#include <cblas.h>
#include "utils.h"
#include "distance.h"
#include "filter/label.h"
#include "filter/selector.h"

// WORKS FOR UPTO 2 BILLION POINTS (as we use INT INSTEAD OF UNSIGNED)

#define PARTSIZE 50000000
#define ALIGNMENT 512
#define Q_BATCH_SIZE (1 << 9)  // dist_matrix bytes ~= Q_BATCH_SIZE * PARTSIZE * 4

using pairIF = std::pair<int, float>;
struct cmpmaxstruct {
  bool operator()(const pairIF &l, const pairIF &r) {
    return l.second < r.second;
  };
};

using maxPQIFCS = std::priority_queue<pairIF, std::vector<pairIF>, cmpmaxstruct>;

static void merge_query_topk(int32_t *global_ids, float *global_dists, size_t k, const int *part_ids,
                             const float *part_dists, uint32_t id_offset) {
  std::vector<std::pair<uint32_t, float>> merged;
  merged.reserve(2 * k);
  for (size_t j = 0; j < k; j++) {
    if (global_ids[j] >= 0) {
      merged.emplace_back((uint32_t) global_ids[j], global_dists[j]);
    }
  }
  for (size_t j = 0; j < k; j++) {
    if (part_ids[j] >= 0) {
      merged.emplace_back(id_offset + (uint32_t) part_ids[j], part_dists[j]);
    }
  }
  std::sort(merged.begin(), merged.end(),
            [](const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b) {
              return a.second < b.second;
            });
  const size_t cnt = std::min(k, merged.size());
  for (size_t j = 0; j < k; j++) {
    if (j < cnt) {
      global_ids[j] = (int32_t) merged[j].first;
      global_dists[j] = merged[j].second;
    } else {
      global_ids[j] = -1;
      global_dists[j] = std::numeric_limits<float>::max();
    }
  }
}

static void init_topk_buffers(int32_t *ids, float *dists, size_t nqueries, size_t k) {
  for (size_t i = 0; i < nqueries * k; i++) {
    ids[i] = -1;
    dists[i] = std::numeric_limits<float>::max();
  }
}

static void init_compute_threads(int num_threads) {
  int nt = num_threads;
  if (nt <= 0) {
    if (const char *env = std::getenv("OMP_NUM_THREADS")) {
      nt = std::max(1, std::atoi(env));
    } else {
      nt = std::max(1, omp_get_max_threads());
    }
  }
  omp_set_num_threads(nt);
  if (std::getenv("OPENBLAS_NUM_THREADS") == nullptr) {
    static std::string blas_threads = std::to_string(nt);
    setenv("OPENBLAS_NUM_THREADS", blas_threads.c_str(), 1);
  }
  std::cout << "OpenMP threads: " << nt
            << ", OPENBLAS_NUM_THREADS: " << (std::getenv("OPENBLAS_NUM_THREADS") ? std::getenv("OPENBLAS_NUM_THREADS")
                                                                                   : "unset")
            << std::endl;
}

void exact_knn(const size_t dim, const size_t k,
               int *const closest_points,         // k * num_queries preallocated, col major, queries columns
               float *const dist_closest_points,  // k * num_queries. preallocated, Dist to corresponding closest_points
               size_t npoints,
               float *points_in,  // Row major
               size_t nqueries,
               float *queries_in,  // Row major
               pipeann::Metric metric = pipeann::Metric::L2) {
  pipeann::Distance<float> *distance = pipeann::get_distance_function<float>(metric);

  const pipeann::DistanceL2Float *l2_dist = nullptr;
  std::vector<float> b_sq;
  if (metric == pipeann::Metric::L2) {
    l2_dist = dynamic_cast<const pipeann::DistanceL2Float *>(distance);
    b_sq.resize(npoints);
    std::cout << "Precomputing base L2 norms for " << npoints << " points..." << std::endl;
#pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < (int64_t) npoints; j++) {
      b_sq[(size_t) j] =
          cblas_sdot((int) dim, points_in + (size_t) j * dim, 1, points_in + (size_t) j * dim, 1);
    }
  }

  std::cout << "Going to compute " << k << " NNs for " << nqueries << " queries over " << npoints << " points in "
            << dim << " dimensions using " << pipeann::get_metric_str(metric) << std::endl;

  size_t q_batch_size = Q_BATCH_SIZE;
  // bulk_compare(queries_batch, batch_size, points, npoints, dim, dist_matrix) produces
  // dist_matrix[q_local * npoints + p], which matches the col-major access dist_matrix[p + q_local * npoints].
  float *dist_matrix = new float[(size_t) q_batch_size * (size_t) npoints];

  for (uint64_t b = 0; b < DIV_ROUND_UP(nqueries, q_batch_size); ++b) {
    int64_t q_b = b * q_batch_size;
    int64_t q_e = ((b + 1) * q_batch_size > nqueries) ? nqueries : (b + 1) * q_batch_size;

    if (l2_dist != nullptr) {
      l2_dist->bulk_compare(queries_in + q_b * dim, (size_t) (q_e - q_b), points_in, npoints, (unsigned) dim,
                            dist_matrix, b_sq.data());
    } else {
      distance->bulk_compare(queries_in + q_b * dim, (size_t) (q_e - q_b), points_in, npoints, (unsigned) dim,
                             dist_matrix);
    }
    std::cout << "Computed distances for queries: [" << q_b << "," << q_e << ")" << std::endl;

#pragma omp parallel for schedule(static)
    for (int64_t q = q_b; q < q_e; q++) {
      maxPQIFCS point_dist;
      for (uint64_t p = 0; p < k; p++)
        point_dist.emplace(p, dist_matrix[(q - q_b) * npoints + p]);
      for (uint64_t p = k; p < npoints; p++) {
        if (point_dist.top().second > dist_matrix[(q - q_b) * npoints + p])
          point_dist.emplace(p, dist_matrix[(q - q_b) * npoints + p]);
        if (point_dist.size() > k)
          point_dist.pop();
      }
      for (int64_t l = 0; l < (int64_t) k; ++l) {
        closest_points[(k - 1 - l) + q * k] = point_dist.top().first;
        dist_closest_points[(k - 1 - l) + q * k] = point_dist.top().second;
        point_dist.pop();
      }
    }
    std::cout << "Computed exact k-NN for queries: [" << q_b << "," << q_e << ")" << std::endl;
  }

  delete[] dist_matrix;
  delete distance;

  if (metric == pipeann::Metric::COSINE) {
    delete[] points_in;
    delete[] queries_in;
  }
}

template<typename T>
inline uint32_t get_base_part_info(const char *filename, size_t &total_npts, size_t &dim) {
  std::ifstream reader(filename, std::ios::binary);
  std::cout << "Reading bin file " << filename << " ...\n";
  int npts_i32, ndims_i32;
  reader.read((char *) &npts_i32, sizeof(int));
  reader.read((char *) &ndims_i32, sizeof(int));
  reader.close();
  total_npts = (size_t) npts_i32;
  dim = (size_t) ndims_i32;
  std::cout << "#pts = " << total_npts << ", #dims = " << dim << std::endl;
  uint32_t num_parts = (total_npts % PARTSIZE) == 0 ? (uint32_t) (total_npts / PARTSIZE)
                                                    : (uint32_t) (total_npts / PARTSIZE) + 1;
  std::cout << "Number of parts: " << num_parts << std::endl;
  return num_parts;
}

// Load an entire bin file (e.g. queries), or one base chunk when total_npts > 0.
template<typename T>
inline void load_bin_as_float(const char *filename, float *&data, size_t &npts, size_t &ndims, int part_num,
                              size_t total_npts = 0, size_t file_dim = 0) {
  std::vector<T> data_vec;
  if (total_npts == 0) {
    pipeann::load_bin<T>(filename, data_vec, npts, ndims, 0);
  } else {
    const size_t start_pt = (size_t) part_num * PARTSIZE;
    if (start_pt >= total_npts) {
      npts = 0;
      ndims = file_dim;
      data = nullptr;
      return;
    }
    npts = std::min((size_t) PARTSIZE, total_npts - start_pt);
    ndims = file_dim;
    const size_t byte_offset = 2 * sizeof(int) + start_pt * file_dim * sizeof(T);
    data_vec.resize(npts * file_dim);
    std::ifstream reader(filename, std::ios::binary);
    if (!reader.is_open()) {
      std::cerr << "Error: cannot open " << filename << std::endl;
      exit(-1);
    }
    reader.seekg((std::streamoff) byte_offset);
    reader.read((char *) data_vec.data(), (std::streamsize) (npts * file_dim * sizeof(T)));
    if (!reader) {
      std::cerr << "Error: failed reading base part " << part_num << " at byte offset " << byte_offset << std::endl;
      exit(-1);
    }
    std::cout << "Loaded base part " << part_num << " [" << start_pt << ", " << start_pt + npts << ") of "
              << total_npts << std::endl;
  }
  pipeann::alloc_aligned((void **) &data, npts * ndims * sizeof(float), ALIGNMENT);
  pipeann::convert_types<T, float>(data_vec.data(), data, npts, ndims);
  std::cout << "Finished converting part data to float." << std::endl;
}

inline void save_groundtruth_as_one_file(const std::string filename, int32_t *data, float *distances, size_t npts,
                                         size_t ndims, uint32_t *tags = nullptr) {
  std::ofstream writer(filename, std::ios::binary | std::ios::out);
  int npts_i32 = (int) npts, ndims_i32 = (int) ndims;
  writer.write((char *) &npts_i32, sizeof(int));
  writer.write((char *) &ndims_i32, sizeof(int));
  std::cout << "Saving truthset in one file (npts, dim, npts*dim id-matrix, "
               "npts*dim dist-matrix) with npts = "
            << npts << ", dim = " << ndims << ", size = " << 2 * npts * ndims * sizeof(unsigned) + 2 * sizeof(int)
            << "B" << std::endl;

  //    data = new T[npts_u64 * ndims_u64];
  writer.write((char *) data, npts * ndims * sizeof(uint32_t));
  writer.write((char *) distances, npts * ndims * sizeof(float));
  if (tags != nullptr) {
    writer.write((char *) tags, npts * ndims * sizeof(uint32_t));
  } else {
    writer.write((char *) data, npts * ndims * sizeof(uint32_t));
  }

  writer.close();
  std::cout << "Finished writing truthset" << std::endl;
}

// Pre-filter: get all base point IDs that match the query's filter criteria
inline std::vector<uint32_t> get_matching_points(pipeann::AbstractSelector *selector, pipeann::SpmatLabel *base_labels,
                                                 pipeann::SpmatLabel *query_labels, uint32_t query_id,
                                                 uint32_t start_id, uint32_t end_id) {
  std::vector<uint32_t> matching_points;

  // Prepare query label buffer
  size_t query_label_size = query_labels->label_size();
  std::vector<char> query_label_buf(query_label_size, 0);
  query_labels->write(query_id, query_label_buf.data());

  // Prepare base label buffer (reused)
  size_t base_label_size = base_labels->label_size();
  std::vector<char> base_label_buf(base_label_size, 0);

  for (uint32_t point_id = start_id; point_id < end_id; point_id++) {
    base_labels->write(point_id, base_label_buf.data());
    if (selector->is_member(point_id, query_label_buf.data(), base_label_buf.data())) {
      matching_points.push_back(point_id);
    }
  }

  return matching_points;
}

// Compute distance between a query and a single point
inline float compute_distance(const float *query, const float *point, size_t dim, pipeann::Distance<float> *distance) {
  return distance->compare(query, point, dim);
}

template<typename T>
int aux_main(int argc, char **argv, pipeann::Metric metric, pipeann::AbstractSelector *selector,
             pipeann::SpmatLabel *base_labels, pipeann::SpmatLabel *query_labels) {
  size_t npoints, nqueries, dim;
  std::string base_file(argv[3]);
  std::string query_file(argv[4]);
  size_t k = atoi(argv[5]);
  std::string gt_file(argv[6]);

  float *base_data;
  float *query_data;

  size_t base_npts, base_dim;
  int num_parts = (int) get_base_part_info<T>(base_file.c_str(), base_npts, base_dim);
  load_bin_as_float<T>(query_file.c_str(), query_data, nqueries, dim, -1);

  if (metric == pipeann::Metric::COSINE) {  // we convert cosine distance as normalized L2/IP distance.
    for (uint64_t i = 0; i < nqueries; i++) {
      pipeann::normalize_data(query_data + i * dim, query_data + i * dim, dim);
    }
  }

  pipeann::Distance<float> *distance = pipeann::get_distance_function<float>(metric);

  int *closest_points = new int[nqueries * k];
  float *dist_closest_points = new float[nqueries * k];
  init_topk_buffers(closest_points, dist_closest_points, nqueries, k);

  for (int p = 0; p < num_parts; p++) {
    size_t start_id = p * PARTSIZE;
    load_bin_as_float<T>(base_file.c_str(), base_data, npoints, dim, p, base_npts, base_dim);
    if (metric == pipeann::Metric::COSINE) {
#pragma omp parallel for schedule(static)
      for (int64_t i = 0; i < (int64_t) npoints; i++) {
        pipeann::normalize_data(base_data + (size_t) i * dim, base_data + (size_t) i * dim, dim);
      }
    }

    if (selector == nullptr) {
      // No filtering: use fast batch KNN
      int *closest_points_part = new int[nqueries * k];
      float *dist_closest_points_part = new float[nqueries * k];

      exact_knn(dim, k, closest_points_part, dist_closest_points_part, npoints, base_data, nqueries, query_data,
                metric);

#pragma omp parallel for schedule(static)
      for (int64_t i = 0; i < (int64_t) nqueries; i++) {
        merge_query_topk(closest_points + (size_t) i * k, dist_closest_points + (size_t) i * k, k,
                         closest_points_part + (size_t) i * k, dist_closest_points_part + (size_t) i * k,
                         (uint32_t) start_id);
      }

      delete[] closest_points_part;
      delete[] dist_closest_points_part;
    } else {
      // Pre-filtering: for each query, first filter points, then compute distances
      std::cout << "Pre-filtering for part " << p << " (points " << start_id << " to " << start_id + npoints << ")"
                << std::endl;

#pragma omp parallel for schedule(dynamic, 1)
      for (int64_t q = 0; q < (int64_t) nqueries; q++) {
        std::vector<uint32_t> matching_points =
            get_matching_points(selector, base_labels, query_labels, (uint32_t) q, (uint32_t) start_id,
                                (uint32_t) (start_id + npoints));

        if (matching_points.empty()) {
          continue;
        }

        std::vector<std::pair<uint32_t, float>> point_dists;
        point_dists.reserve(matching_points.size());

        for (uint32_t point_id : matching_points) {
          uint32_t local_id = point_id - (uint32_t) start_id;
          float dist = compute_distance(query_data + (size_t) q * dim, base_data + local_id * dim, dim, distance);
          point_dists.emplace_back(point_id, dist);
        }

        std::sort(point_dists.begin(), point_dists.end(),
                  [](const std::pair<uint32_t, float> &a, const std::pair<uint32_t, float> &b) {
                    return a.second < b.second;
                  });
        const size_t part_k = std::min(k, point_dists.size());
        std::vector<int> part_ids(k, -1);
        std::vector<float> part_dists(k, std::numeric_limits<float>::max());
        for (size_t j = 0; j < part_k; j++) {
          part_ids[j] = (int) point_dists[j].first;
          part_dists[j] = point_dists[j].second;
        }
        merge_query_topk(closest_points + (size_t) q * k, dist_closest_points + (size_t) q * k, k, part_ids.data(),
                         part_dists.data(), 0);
      }

      std::cout << "Finished pre-filtering for part " << p << std::endl;
    }

    pipeann::aligned_free(base_data);
    std::cout << "Merged global top-" << k << " after part " << p << std::endl;
  }

  if (metric == pipeann::Metric::INNER_PRODUCT) {
#pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < (int64_t) nqueries * (int64_t) k; i++) {
      if (closest_points[i] >= 0) {
        dist_closest_points[i] = -dist_closest_points[i];
      }
    }
  }

#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < (int64_t) nqueries; i++) {
    size_t valid_count = 0;
    for (uint64_t j = 0; j < k; j++) {
      if (closest_points[(size_t) i * k + j] >= 0) {
        ++valid_count;
      }
    }
    if (valid_count < k) {
#pragma omp critical
      std::cout << "WARNING: Query " << i << " only found " << valid_count << " matching results (requested " << k
                << ")" << std::endl;
    }
  }

  uint32_t *tags = nullptr;
  if (std::string(argv[7]) != std::string("null")) {
    std::cout << "Loading tags from " << argv[7] << "\n";
    tags = new uint32_t[nqueries * k];
    uint32_t *all_tags;
    std::string tag_file = std::string(argv[7]);
    size_t tag_pts, tag_dim;
    pipeann::load_bin(tag_file, all_tags, tag_pts, tag_dim);

    std::cout << "Loaded tags for " << tag_pts << " points.\n";
    for (uint64_t i = 0; i < nqueries * k; i++) {
      if (closest_points[i] >= 0) {
        tags[i] = all_tags[closest_points[i]];
      } else {
        tags[i] = 0;  // Invalid tag for padded entries
      }
    }
  }

  save_groundtruth_as_one_file(gt_file, closest_points, dist_closest_points, nqueries, k, tags);
  pipeann::aligned_free(query_data);
  delete[] closest_points;
  delete[] dist_closest_points;
  delete distance;
  if (tags != nullptr) {
    delete[] tags;
  }

  return 0;
}

int main(int argc, char **argv) {
  if (argc < 9) {
    std::cerr << "Usage: " << argv[0]
              << " <int8/uint8/float> <distance function (l2/cosine/mips)> <base bin file> <query bin file> <K> "
                 "<output-truthset-file> <tag_file (null if none)> <label_type (spmat/null)> <selector_type "
                 "(range/intersect/subset)> <base_label_file.spmat> <query_label_file.spmat> [num_threads (0=auto)]"
              << std::endl;
    return -1;
  }

  int num_threads = 0;
  if (argc >= 13) {
    num_threads = std::atoi(argv[12]);
  }
  init_compute_threads(num_threads);

  std::string data_type(argv[1]);
  std::string dist_fn(argv[2]);

  pipeann::Metric metric = pipeann::get_metric(dist_fn);

  std::string label_type(argv[8]);

  // Get selector
  pipeann::AbstractSelector *selector = nullptr;
  pipeann::SpmatLabel *base_labels = nullptr;
  pipeann::SpmatLabel *query_labels = nullptr;

  if (label_type != "null") {
    selector = pipeann::get_selector<float>(argv[9]);
    if (selector == nullptr) {
      std::cerr << "Unknown selector type: " << argv[9] << ". Use range/intersect/subset." << std::endl;
      return -1;
    }
    base_labels = new pipeann::SpmatLabel(argv[10]);
    query_labels = new pipeann::SpmatLabel(argv[11]);
    std::cout << "Loaded base labels: " << base_labels->labels_.size() << " vectors" << std::endl;
    std::cout << "Loaded query labels: " << query_labels->labels_.size() << " queries" << std::endl;
  }

  if (data_type == std::string("float")) {
    aux_main<float>(argc, argv, metric, selector, base_labels, query_labels);
  } else if (data_type == std::string("int8")) {
    aux_main<int8_t>(argc, argv, metric, selector, base_labels, query_labels);
  } else if (data_type == std::string("uint8")) {
    aux_main<uint8_t>(argc, argv, metric, selector, base_labels, query_labels);
  } else {
    std::cout << "Unsupported type. float, int8 and uint8 types are supported." << std::endl;
    return -1;
  }

  return 0;
}
