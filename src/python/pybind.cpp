#include "pyindex.h"

PYBIND11_MODULE(C, m) {
  m.doc() = "PipeANN";
  m.attr("__version__") = "dev";

  py::enum_<pipeann::Metric>(m, "Metric")
      .value("L2", pipeann::Metric::L2)
      .value("INNER_PRODUCT", pipeann::Metric::INNER_PRODUCT)
      .value("COSINE", pipeann::Metric::COSINE)
      .export_values();

  py::class_<PyIndexInterface>(m, "PyIndex")
      .def(py::init<py::dict>(), py::arg("params"))
      .def("load", &PyIndexInterface::load, py::arg("index_prefix"))
      .def("save", &PyIndexInterface::save, py::arg("index_prefix"))
      .def("build", &PyIndexInterface::build, py::arg("data_path"), py::arg("index_prefix"),
           py::arg("tag_file") = nullptr, py::arg("build_mem_index") = false, py::arg("max_nbrs") = 0,
           py::arg("build_L") = 0, py::arg("PQ_bytes") = 32, py::arg("memory_use_GB") = 0)
      .def("search", &PyIndexInterface::search, py::arg("queries"), py::arg("topk"), py::arg("L"))
      .def("add", &PyIndexInterface::add, py::arg("vectors"), py::arg("tags"))
      .def("remove", &PyIndexInterface::remove, py::arg("tags"))
      .def("set_index_prefix", &PyIndexInterface::set_index_prefix, py::arg("index_prefix"))
      .def("omp_set_num_threads", &PyIndexInterface::omp_set_num_threads, py::arg("num_threads"))
      .def("__repr__", &PyIndexInterface::to_string);
}