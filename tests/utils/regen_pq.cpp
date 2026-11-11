// Regenerate *_pq_pivots.bin and *_pq_compressed.bin for an existing index prefix.
#include "nbr/pq_nbr.h"
#include "utils.h"
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " <float|int8|uint8> <data.bin> <index_prefix> <pq_bytes>\n";
    return 1;
  }
  std::string type = argv[1];
  std::string data_bin = argv[2];
  std::string prefix = argv[3];
  uint32_t pq_bytes = (uint32_t) std::atoi(argv[4]);

  auto run = [&](auto *nbr) {
    nbr->build(prefix, data_bin, pq_bytes);
    delete nbr;
  };

  if (type == "float") {
    run(new pipeann::PQNeighbor<float>(pipeann::Metric::L2));
  } else if (type == "int8") {
    run(new pipeann::PQNeighbor<int8_t>(pipeann::Metric::L2));
  } else if (type == "uint8") {
    run(new pipeann::PQNeighbor<uint8_t>(pipeann::Metric::L2));
  } else {
    std::cerr << "Unsupported type\n";
    return 1;
  }
  std::cout << "PQ regenerated under prefix: " << prefix << std::endl;
  return 0;
}
