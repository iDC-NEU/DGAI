#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <filesystem>
#include <mutex>
#include <random>
#include <shared_mutex>
#include "index.h"
#include "linux_aligned_file_reader.h"
#include "nbr/nbr.h"
#include "utils/partition.h"
#include "utils/index_build_utils.h"
#include "ssd_index.h"
#include "utils.h"
#include <sys/sysinfo.h>
#include <gperftools/malloc_extension.h>

namespace py = pybind11;

struct PyIndexParams {
  py::dtype data_type = py::dtype::of<float>();
  py::dtype tag_type = py::dtype::of<uint32_t>();
  uint32_t data_dim = 0;
  pipeann::Metric metric = pipeann::Metric::L2;
  uint32_t max_nthreads = 32;
  uint32_t sampled_nbrs_for_delete = 20;
  uint32_t build_threshold = 100000;

  PyIndexParams(py::dtype data_type = py::dtype::of<float>(), py::dtype tag_type = py::dtype::of<uint32_t>(),
                uint32_t data_dim = 0, pipeann::Metric metric = pipeann::Metric::L2, uint32_t max_nthreads = 32,
                uint32_t sampled_nbrs_for_delete = 20, uint32_t build_threshold = 100000)
      : data_type(data_type), tag_type(tag_type), data_dim(data_dim), metric(metric), max_nthreads(max_nthreads),
        sampled_nbrs_for_delete(sampled_nbrs_for_delete), build_threshold(build_threshold) {
  }
};

class BasePyIndex {
 public:
  BasePyIndex() = default;
  virtual ~BasePyIndex() = default;
};

template<class T>
class PyIndex : public BasePyIndex {
  static constexpr float kMemIndexP = 0.01;
  using TagT = uint32_t;

 public:
  PyIndex() = delete;

  explicit PyIndex(PyIndexParams params) : params_(std::move(params)) {
    reader_.reset(new LinuxAlignedFileReader());
    nbr_handler_ = new pipeann::PQNeighbor<T>(params_.metric);
    mem_index_.reset(new pipeann::Index<T, TagT>(params_.metric, params_.data_dim));
    disk_index_.reset(new pipeann::SSDIndex<T, TagT>(params_.metric, reader_, nbr_handler_, true, nullptr));

    mem_index_params_.set(32, 64, 750, 1.2, params_.max_nthreads, true);
    cur_index_prefix_ = "./test";
    // rand value for updating in-memory index.
    std::random_device rd;
    gen = std::mt19937(rd());
  }

  float rand_uniform() {
    std::uniform_real_distribution<float> dist(0, 1);
    return dist(gen);
  }

  ~PyIndex() {
  }

  void load(const std::string &index_prefix) {
    auto mem_index_path = index_prefix + "_mem.index";
    if (file_exists(mem_index_path)) {
      use_mem_index_for_disk_index_ = true;
      mem_index_->load(mem_index_path.c_str());
    } else {
      use_mem_index_for_disk_index_ = false;
    }

    auto disk_index_file = index_prefix + "_disk.index";
    if (file_exists(disk_index_file)) {
      use_disk_index_ = true;
      disk_index_->load(index_prefix.c_str(), params_.max_nthreads, false);
      if (use_mem_index_for_disk_index_) {
        // mem_index is disk_index's navigation graph.
        disk_index_->mem_index_.reset(mem_index_.get());
      }
      mem_L = use_mem_index_for_disk_index_ ? 10 : 0;
    } else {
      use_disk_index_ = false;
    }

    cur_index_prefix_ = index_prefix;
  }

  void transform_mem_index_to_disk_index() {
    auto mu = std::lock_guard<std::shared_mutex>(save_mu_);
    LOG(INFO) << "Transform memory index to disk index.";
    mem_index_->save_data(cur_index_prefix_ + "_mem_data.bin");
    LOG(INFO) << "Memory index data saved to " << (cur_index_prefix_ + "_mem_data.bin");
    mem_index_->save_tags(cur_index_prefix_ + "_disk.index.tags");
    LOG(INFO) << "Memory index tags saved to " << (cur_index_prefix_ + "_disk.index.tags");
    // re-build
    this->build(cur_index_prefix_ + "_mem_data.bin", cur_index_prefix_,
                (cur_index_prefix_ + "_disk.index.tags").c_str(), false);
    this->load(cur_index_prefix_);  // here sets use_disk_index_ to true.
    LOG(INFO) << "Transform memory index to disk index done.";
  }

  // Use file path to build. For easier interface, refer to PyIndexInterface.
  void build(const std::string &data_path, const std::string &index_prefix, const char *tag_file = nullptr,
             bool build_mem_index = false, uint32_t max_nbrs = 0, uint32_t build_L = 0, uint32_t PQ_bytes = 32,
             uint32_t memory_use_GB = 0) {
    // automatically configure max_nbrs.
    size_t nr = 0, nc = 0;
    pipeann::get_bin_metadata(data_path, nr, nc);
    int nr_log10 = std::min(9, (int) std::ceil(std::log10(nr)));

    if (max_nbrs == 0) {
      uint32_t nr_nbr_arr[] = {64, 64, 64, 64, 64, 64, /* 1M */ 64, /* 10M */ 64, /* 100M */ 96, /* 1B */ 128};
      max_nbrs = nr_nbr_arr[nr_log10];
      LOG(INFO) << "Dataset contains " << nr << " points. Setting max_nbrs to " << max_nbrs;
    }

    if (build_L == 0) {
      build_L = max_nbrs + 32;
    }

    if (memory_use_GB == 0) {
      struct sysinfo info;
      sysinfo(&info);
      memory_use_GB = info.totalram / (1024 * 1024 * 1024) * 3 / 4;
      LOG(INFO) << "Memory use not specified. Using 75% of total memory: " << memory_use_GB << "GB";
    }
    pipeann::build_disk_index<T, TagT>(data_path.c_str(), index_prefix.c_str(), max_nbrs, build_L, memory_use_GB,
                                       params_.max_nthreads, PQ_bytes, params_.metric, tag_file, nbr_handler_,
                                       nullptr);  // create tag later.

    if (build_mem_index) {
      build_mem(data_path, index_prefix);
    }
  }

  void set_index_prefix(const std::string &index_prefix) {
    cur_index_prefix_ = index_prefix;
  }

  void omp_set_num_threads(uint32_t num_threads) {
    ::omp_set_num_threads(num_threads);
    this->n_threads = num_threads;
  }

  void build_mem(const std::string &data_path, const std::string &index_prefix) {
    // sample rate 0.01
    std::string sample_prefix = index_prefix + "_mem_sample";
    gen_random_slice<T>(data_path, sample_prefix, kMemIndexP);

    std::string sample_data_bin = sample_prefix + "_data.bin";
    uint64_t data_num, data_dim;
    pipeann::get_bin_metadata(sample_data_bin, data_num, data_dim);

    std::string sample_id_bin = sample_prefix + "_ids.bin";
    std::ifstream reader;
    reader.open(sample_id_bin, std::ios::binary);
    reader.seekg(2 * sizeof(uint32_t), std::ios::beg);
    uint32_t tags_size = data_num * data_dim;
    std::vector<TagT> tags(data_num);
    reader.read((char *) tags.data(), tags_size * sizeof(uint32_t));
    reader.close();

    auto s = std::chrono::high_resolution_clock::now();
    mem_index_.reset(new pipeann::Index<T, TagT>(params_.metric, data_dim));
    // We should normalize_cosine here, as data is not pre-normalized.
    mem_index_->build(sample_data_bin.c_str(), data_num, mem_index_params_, tags);
    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

    LOG(INFO) << "Finish building memory index, indexing time: " << diff.count() << "\n";
    std::string save_path = index_prefix + "_mem.index";
    mem_index_->save(save_path.c_str());
  }

  std::tuple<py::array_t<TagT>, py::array_t<float>> search(py::array_t<T> &queries, uint32_t topk, uint32_t L) {
    auto mu = std::shared_lock<std::shared_mutex>(save_mu_);

    auto queries_buf = queries.request();
    auto n_queries = queries_buf.shape[0];
    auto dim = queries_buf.shape[1];

    auto ret_ids = py::array_t<TagT>({n_queries, (py::ssize_t) topk});
    auto ret_dists = py::array_t<float>({n_queries, (py::ssize_t) topk});

    auto *queries_p = static_cast<T *>(queries_buf.ptr);
    auto *ret_ids_p = static_cast<TagT *>(ret_ids.request().ptr);
    auto *ret_dists_p = static_cast<float *>(ret_dists.request().ptr);

#pragma omp parallel for schedule(dynamic) num_threads(n_threads)
    for (uint32_t i = 0; i < n_queries; i++) {
      T *query_p = queries_p + i * dim;
      std::vector<TagT> tags(L);
      std::vector<float> distances(L);
      if (use_disk_index_) {
        disk_index_->pipe_search(query_p, L, mem_L, L, tags.data(), distances.data(), 32);
      } else {
        mem_index_->search_with_tags(query_p, L, L, tags.data(), distances.data());
      }

      // filter out deleted nodes.
      TagT *ret_id_p = ret_ids_p + i * topk;
      float *ret_dist_p = ret_dists_p + i * topk;
      size_t pos = 0;
      for (size_t j = 0; j < L && pos < topk; j++) {
        if (this->deleted_nodes_set_.find(tags[j]) == this->deleted_nodes_set_.end()) {
          ret_id_p[pos] = tags[j];
          ret_dist_p[pos] = distances[j];
          pos++;
        }
      }
    }
    return std::make_tuple(ret_ids, ret_dists);
  }

  // bulk add.
  void add(py::array_t<T> &vectors, py::array_t<TagT> &tags) {
    save_mu_.lock_shared();
    auto *vectors_p = static_cast<T *>(vectors.request().ptr);
    auto *tags_p = static_cast<TagT *>(tags.request().ptr);
    auto n_vectors = vectors.shape(0);

#pragma omp parallel for schedule(dynamic) num_threads(n_threads)
    for (uint32_t i = 0; i < n_vectors; i++) {
      auto *vector_p = vectors_p + i * params_.data_dim;
      auto *tag_p = tags_p + i;
      do_insert(vector_p, *tag_p);
    }
    save_mu_.unlock_shared();

    if (!use_disk_index_ && mem_index_->get_num_points() > params_.build_threshold) {
      transform_mem_index_to_disk_index();
    }
  }

  void do_insert(T *point_p, TagT tag) {
    if (use_disk_index_) {
      int target_id = disk_index_->insert_in_place(point_p, tag, &this->deleted_nodes_set_);
      if (use_mem_index_for_disk_index_ && rand_uniform() <= kMemIndexP) {  // probably insert into the memory index.
        mem_index_->insert_point(point_p, mem_index_params_, target_id);
      }
    } else {
      mem_index_->insert_point(point_p, mem_index_params_, tag);
    }
  }

  void remove(py::array_t<TagT> &tags) {
    auto mu = std::shared_lock<std::shared_mutex>(save_mu_);
    auto tags_buf = tags.request();
    TagT *tags_ptr = static_cast<TagT *>(tags_buf.ptr);
    // Cannot parallel, as deleted_nodes_ is not thread_safe.
    // This process is fast, so parallel is not mandatory.
    for (py::ssize_t i = 0; i < tags_buf.shape[0]; i++) {
      TagT tag = tags_ptr[i];
      if (deleted_nodes_set_.find(tag) == deleted_nodes_set_.end()) {
        deleted_nodes_set_.insert(tag);
        deleted_nodes_.push_back(tag);
      }
      mem_index_->lazy_delete(tag);
    }
  }

  std::string to_string() const {
    return std::string("PyIndex with data type ") + params_.data_type.char_() + " and tag type " +
           params_.tag_type.char_();
  }

  bool save(const std::string &index_prefix /* requires double-version. */) {
    auto mu = std::lock_guard<std::shared_mutex>(save_mu_);

    if (cur_index_prefix_ == "") {
      LOG(ERROR) << "Current index prefix is empty. Cannot save.";
      return false;
    }

    std::string save_index_prefix = index_prefix;
    if (cur_index_prefix_ == index_prefix) {
      save_index_prefix = index_prefix + "_v2";  // double-version.
      LOG(INFO) << "Saving to the same index prefix. Using double-version: " << save_index_prefix;
    }

    if (use_disk_index_) {
      disk_index_->merge_deletes(cur_index_prefix_, save_index_prefix, deleted_nodes_, deleted_nodes_set_, n_threads,
                                 params_.sampled_nbrs_for_delete);
      disk_index_->reload(save_index_prefix.c_str(), 1);  // reload the newest index.

      if (cur_index_prefix_ == index_prefix) {
        LOG(INFO) << "Copying index from " << save_index_prefix << " to " << index_prefix;
        disk_index_->copy_index(save_index_prefix, index_prefix);  // copy to the specified prefix.
        disk_index_->reload(index_prefix.c_str(), 1);
      }  // only change when using double-version.

      cur_index_prefix_ = index_prefix;
    }

    if (!use_disk_index_ || use_mem_index_for_disk_index_) {
      // if not using disk index, or using mem index for disk index, save mem index.
      mem_index_->save((index_prefix + "_mem.index").c_str());
    }
    MallocExtension::instance()->ReleaseFreeMemory();  // Return free list to OS.
    return true;
  }

 private:
  bool use_disk_index_ = false;
  bool use_mem_index_for_disk_index_ = false;
  std::shared_ptr<AlignedFileReader> reader_;
  pipeann::AbstractNeighbor<T> *nbr_handler_;
  std::shared_mutex save_mu_;  // save mutex.
  std::string data_path_;
  std::string cur_index_prefix_;
  PyIndexParams params_;
  std::vector<TagT> deleted_nodes_;
  tsl::robin_set<TagT> deleted_nodes_set_;  // copy of deleted nodes.

  // if vectors are less than the threshold, use mem index instead.
  std::mt19937 gen;
  pipeann::IndexBuildParameters mem_index_params_;
  std::shared_ptr<pipeann::Index<T, TagT>> mem_index_;
  std::shared_ptr<pipeann::SSDIndex<T, TagT>> disk_index_;
  uint32_t mem_L = 0;      // memory search L in pipe search.
  uint32_t n_threads = 1;  // number of threads to use.
};

class PyIndexInterface {
 public:
  PyIndexInterface() = delete;
  explicit PyIndexInterface(py::dict params);

  // lifecycle and I/O
  void load(const std::string &index_prefix);
  bool save(const std::string &index_prefix);

  // build
  void build(const std::string &data_path, const std::string &index_prefix, const char *tag_file = nullptr,
             bool build_mem_index = false, uint32_t max_nbrs = 0, uint32_t build_L = 0, uint32_t PQ_bytes = 32,
             uint32_t memory_use_GB = 0);

  // updates and queries
  std::tuple<py::array, py::array> search(py::array &queries, uint32_t topk, uint32_t L);
  void add(py::array &vectors, py::array &tags);
  void remove(py::array &tags);
  void set_index_prefix(const std::string &index_prefix);
  void omp_set_num_threads(uint32_t num_threads);

  std::string to_string() const;

  // public to allow helpers in translation unit to reference it
  enum class DataType { F32, U8, I8 };

 private:
  template<typename T>
  PyIndex<T> *get() const {
    return dynamic_cast<PyIndex<T> *>(impl_.get());
  }

  static PyIndexParams parse_params(py::dict params, DataType &dt_out);

 private:
  DataType dtype_;
  PyIndexParams params_;
  std::unique_ptr<BasePyIndex> impl_;
};