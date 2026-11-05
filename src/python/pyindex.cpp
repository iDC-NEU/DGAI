#include "pyindex.h"
#include <pybind11/stl.h>

using pipeann::Metric;

// Removed dtype_from_py function as its logic is now in parse_params.

PyIndexParams PyIndexInterface::parse_params(py::dict params, DataType &dt_out) {
  PyIndexParams ip;
  // data_type
  if (params.contains("data_type")) {
    py::dtype dt = params["data_type"].cast<py::dtype>();
    ip.data_type = dt;
    if (dt.is(py::dtype::of<float>()))
      dt_out = DataType::F32;
    else if (dt.is(py::dtype::of<uint8_t>()))
      dt_out = DataType::U8;
    else if (dt.is(py::dtype::of<int8_t>()))
      dt_out = DataType::I8;
    else
      throw std::runtime_error("Unsupported data dtype; expected float32, uint8, or int8");
  } else {
    ip.data_type = py::dtype::of<float>();
    dt_out = DataType::F32;
  }
  // tag_type (currently fixed to uint32)
  if (params.contains("tag_type")) {
    ip.tag_type = params["tag_type"].cast<py::dtype>();
  } else {
    ip.tag_type = py::dtype::of<uint32_t>();
  }
  // data_dim
  if (params.contains("data_dim"))
    ip.data_dim = params["data_dim"].cast<uint32_t>();
  // metric
  if (params.contains("metric"))
    ip.metric = params["metric"].cast<Metric>();
  // threads
  if (params.contains("max_nthreads"))
    ip.max_nthreads = params["max_nthreads"].cast<uint32_t>();
  if (params.contains("sampled_nbrs_for_delete"))
    ip.sampled_nbrs_for_delete = params["sampled_nbrs_for_delete"].cast<uint32_t>();
  if (params.contains("build_threshold"))
    ip.build_threshold = params["build_threshold"].cast<uint32_t>();
  return ip;
}

PyIndexInterface::PyIndexInterface(py::dict params) {
  params_ = parse_params(params, dtype_);
  switch (dtype_) {
    case DataType::F32:
      impl_.reset(new PyIndex<float>(params_));
      break;
    case DataType::U8:
      impl_.reset(new PyIndex<uint8_t>(params_));
      break;
    case DataType::I8:
      impl_.reset(new PyIndex<int8_t>(params_));
      break;
  }
}

void PyIndexInterface::load(const std::string &index_prefix) {
  if (auto *p = get<float>())
    return p->load(index_prefix), void();
  if (auto *p = get<uint8_t>())
    return p->load(index_prefix), void();
  if (auto *p = get<int8_t>())
    return p->load(index_prefix), void();
  throw std::runtime_error("Invalid underlying index");
}

bool PyIndexInterface::save(const std::string &index_prefix) {
  if (auto *p = get<float>())
    return p->save(index_prefix);
  if (auto *p = get<uint8_t>())
    return p->save(index_prefix);
  if (auto *p = get<int8_t>())
    return p->save(index_prefix);
  throw std::runtime_error("Invalid underlying index");
}

void PyIndexInterface::build(const std::string &data_path, const std::string &index_prefix, const char *tag_file,
                             bool build_mem_index, uint32_t max_nbrs, uint32_t build_L, uint32_t PQ_bytes,
                             uint32_t memory_use_GB) {
  if (auto *p = get<float>())
    return p->build(data_path, index_prefix, tag_file, build_mem_index, max_nbrs, build_L, PQ_bytes, memory_use_GB);
  if (auto *p = get<uint8_t>())
    return p->build(data_path, index_prefix, tag_file, build_mem_index, max_nbrs, build_L, PQ_bytes, memory_use_GB);
  if (auto *p = get<int8_t>())
    return p->build(data_path, index_prefix, tag_file, build_mem_index, max_nbrs, build_L, PQ_bytes, memory_use_GB);
  throw std::runtime_error("Invalid underlying index");
}

std::tuple<py::array, py::array> PyIndexInterface::search(py::array &queries, uint32_t topk, uint32_t L) {
  // Ensure contiguous and correct dtype
  if (dtype_ == DataType::F32) {
    auto q = py::array_t<float>(queries);
    auto res = get<float>()->search(q, topk, L);
    auto ids = std::get<0>(res);
    auto dists = std::get<1>(res);
    return std::make_tuple(ids, dists);
  }
  if (dtype_ == DataType::U8) {
    auto q = py::array_t<uint8_t>(queries);
    auto res = get<uint8_t>()->search(q, topk, L);
    auto ids = std::get<0>(res);
    auto dists = std::get<1>(res);
    return std::make_tuple(ids, dists);
  }
  if (dtype_ == DataType::I8) {
    auto q = py::array_t<int8_t>(queries);
    auto res = get<int8_t>()->search(q, topk, L);
    auto ids = std::get<0>(res);
    auto dists = std::get<1>(res);
    return std::make_tuple(ids, dists);
  }
  throw std::runtime_error("Invalid underlying index");
}

void PyIndexInterface::add(py::array &vectors, py::array &tags) {
  if (dtype_ == DataType::F32) {
    auto v = py::array_t<float>(vectors);
    auto t = py::array_t<uint32_t>(tags);
    return get<float>()->add(v, t);
  }
  if (dtype_ == DataType::U8) {
    auto v = py::array_t<uint8_t>(vectors);
    auto t = py::array_t<uint32_t>(tags);
    return get<uint8_t>()->add(v, t);
  }
  if (dtype_ == DataType::I8) {
    auto v = py::array_t<int8_t>(vectors);
    auto t = py::array_t<uint32_t>(tags);
    return get<int8_t>()->add(v, t);
  }
  throw std::runtime_error("Invalid underlying index");
}

void PyIndexInterface::remove(py::array &tags) {
  auto t = py::array_t<uint32_t>(tags);
  if (auto *p = get<float>())
    return p->remove(t), void();
  if (auto *p = get<uint8_t>())
    return p->remove(t), void();
  if (auto *p = get<int8_t>())
    return p->remove(t), void();
  throw std::runtime_error("Invalid underlying index");
}

void PyIndexInterface::set_index_prefix(const std::string &index_prefix) {
  if (auto *p = get<float>())
    return p->set_index_prefix(index_prefix), void();
  if (auto *p = get<uint8_t>())
    return p->set_index_prefix(index_prefix), void();
  if (auto *p = get<int8_t>())
    return p->set_index_prefix(index_prefix), void();
  throw std::runtime_error("Invalid underlying index");
}

void PyIndexInterface::omp_set_num_threads(uint32_t num_threads) {
  if (auto *p = get<float>())
    return p->omp_set_num_threads(num_threads), void();
  if (auto *p = get<uint8_t>())
    return p->omp_set_num_threads(num_threads), void();
  if (auto *p = get<int8_t>())
    return p->omp_set_num_threads(num_threads), void();
  throw std::runtime_error("Invalid underlying index");
}

std::string PyIndexInterface::to_string() const {
  if (auto *p = dynamic_cast<PyIndex<float> *>(impl_.get()))
    return p->to_string();
  if (auto *p = dynamic_cast<PyIndex<uint8_t> *>(impl_.get()))
    return p->to_string();
  if (auto *p = dynamic_cast<PyIndex<int8_t> *>(impl_.get()))
    return p->to_string();
  return "PyIndexInterface";
}
