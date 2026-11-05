#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "utils.h"
#include "ssd_index_defs.h"

namespace {

template<typename T>
pipeann::DiskNode<T> disk_node_at(char *buf, uint64_t loc, const pipeann::SSDIndexMetadata<T> &meta) {
  return pipeann::DiskNode<T>(buf, static_cast<uint32_t>(loc), meta);
}

template<typename T>
void copy_graph_node(char *dst, const pipeann::DiskNode<T> &node, uint64_t max_graph_node_len, uint64_t range,
                     uint64_t label_size) {
  std::memset(dst, 0, max_graph_node_len);
  auto *nnbrs = reinterpret_cast<uint32_t *>(dst);
  *nnbrs = node.nnbrs;
  std::memcpy(dst + sizeof(uint32_t), node.nbrs, node.nnbrs * sizeof(uint32_t));
  if (label_size > 0)
    std::memcpy(dst + sizeof(uint32_t) + range * sizeof(uint32_t), node.labels, label_size);
}

template<typename T>
int extract_graph_pass(int graph_in_fd, int graph_out_fd, const pipeann::SSDIndexMetadata<T> &meta,
                       uint64_t total_loc, uint64_t max_graph_node_len, uint64_t bytes_per_write_graph,
                       uint64_t nodes_per_write_graph, uint64_t range, uint64_t label_size) {
  const uint64_t max_node_len = meta.max_node_len;
  const uint64_t nnodes_per_sector = meta.nnodes_per_sector;
  const uint64_t bytes_per_read = nnodes_per_sector > 0 ? SECTOR_LEN : ROUND_UP(max_node_len, SECTOR_LEN);
  const uint64_t nodes_per_read = nnodes_per_sector > 0 ? nnodes_per_sector : 1;

  std::vector<char> read_buf(bytes_per_read);
  std::vector<char> graph_buf(bytes_per_write_graph, 0);
  off_t out_off = 0;

  std::cout << "Pass 1/2: extracting graph (sequential read/write, " << total_loc << " locations)" << std::endl;

  for (uint64_t loc = 0; loc < total_loc;) {
    uint64_t nodes_this_read = std::min(nodes_per_read, total_loc - loc);
    uint64_t sec_no = 1 + (nnodes_per_sector > 0 ? loc / nnodes_per_sector
                                                   : loc * DIV_ROUND_UP(max_node_len, SECTOR_LEN));
    if (pread(graph_in_fd, read_buf.data(), bytes_per_read, (off_t) sec_no * SECTOR_LEN) != (ssize_t) bytes_per_read) {
      std::cerr << "Error: pread failed at graph loc " << loc << std::endl;
      return -1;
    }

    for (uint64_t j = 0; j < nodes_this_read; ++j, ++loc) {
      uint64_t slot = loc % nodes_per_write_graph;
      if (slot == 0 && loc > 0) {
        if (pwrite(graph_out_fd, graph_buf.data(), bytes_per_write_graph, out_off) != (ssize_t) bytes_per_write_graph) {
          std::cerr << "Error: pwrite failed at graph offset " << out_off << std::endl;
          return -1;
        }
        out_off += (off_t) bytes_per_write_graph;
        std::fill(graph_buf.begin(), graph_buf.end(), 0);
      }

      auto node = disk_node_at<T>(read_buf.data(), loc, meta);
      copy_graph_node<T>(graph_buf.data() + slot * max_graph_node_len, node, max_graph_node_len, range, label_size);
    }

    if (loc % 500000 == 0 || loc == total_loc)
      std::cout << "Graph: " << loc << "/" << total_loc << std::endl;
  }

  if (pwrite(graph_out_fd, graph_buf.data(), bytes_per_write_graph, out_off) != (ssize_t) bytes_per_write_graph) {
    std::cerr << "Error: pwrite failed at final graph sector" << std::endl;
    return -1;
  }
  return 0;
}

template<typename T>
int extract_vector_pass(int vec_in_fd, int vec_out_fd, const pipeann::SSDIndexMetadata<T> &meta, uint64_t npoints,
                        uint64_t vec_data_len, uint64_t bytes_per_write_vec, uint64_t nodes_per_write_vec) {
  const uint64_t max_node_len = meta.max_node_len;
  const uint64_t nnodes_per_sector = meta.nnodes_per_sector;
  const uint64_t bytes_per_read = nnodes_per_sector > 0 ? SECTOR_LEN : ROUND_UP(max_node_len, SECTOR_LEN);
  const uint64_t nodes_per_read = nnodes_per_sector > 0 ? nnodes_per_sector : 1;

  std::vector<char> read_buf(bytes_per_read);
  std::vector<char> vec_buf(bytes_per_write_vec, 0);
  off_t out_off = 0;

  std::cout << "Pass 2/2: extracting vectors (sequential read/write, " << npoints << " points)" << std::endl;

  for (uint64_t id = 0; id < npoints;) {
    uint64_t nodes_this_read = std::min(nodes_per_read, npoints - id);
    uint64_t sec_no = 1 + (nnodes_per_sector > 0 ? id / nnodes_per_sector
                                                   : id * DIV_ROUND_UP(max_node_len, SECTOR_LEN));
    if (pread(vec_in_fd, read_buf.data(), bytes_per_read, (off_t) sec_no * SECTOR_LEN) != (ssize_t) bytes_per_read) {
      std::cerr << "Error: pread failed at vector id " << id << std::endl;
      return -1;
    }

    for (uint64_t j = 0; j < nodes_this_read; ++j, ++id) {
      uint64_t slot = id % nodes_per_write_vec;
      if (slot == 0 && id > 0) {
        if (pwrite(vec_out_fd, vec_buf.data(), bytes_per_write_vec, out_off) != (ssize_t) bytes_per_write_vec) {
          std::cerr << "Error: pwrite failed at vector offset " << out_off << std::endl;
          return -1;
        }
        out_off += (off_t) bytes_per_write_vec;
        std::fill(vec_buf.begin(), vec_buf.end(), 0);
      }

      auto node = disk_node_at<T>(read_buf.data(), id, meta);
      std::memcpy(vec_buf.data() + slot * vec_data_len, node.coords, vec_data_len);
    }

    if (id % 500000 == 0 || id == npoints)
      std::cout << "Vector: " << id << "/" << npoints << std::endl;
  }

  if (pwrite(vec_out_fd, vec_buf.data(), bytes_per_write_vec, out_off) != (ssize_t) bytes_per_write_vec) {
    std::cerr << "Error: pwrite failed at final vector sector" << std::endl;
    return -1;
  }
  return 0;
}

}  // namespace

template<typename T>
int split_disk_index(const std::string &index_prefix) {
  std::string index_file = index_prefix + "_disk.index";
  std::string graph_file = index_prefix + "_disk.graph";
  std::string vector_file = index_prefix + "_disk.vector";

  pipeann::SSDIndexMetadata<T> meta;
  meta.load_from_disk_index(index_file);
  meta.print();

  if (meta.range > 1000000 || meta.label_size > 1000000 || meta.data_dim > 100000) {
    std::cerr << "Error: metadata appears corrupted (range=" << meta.range
              << ", label_size=" << meta.label_size << ", data_dim=" << meta.data_dim << ")." << std::endl;
    return -1;
  }

  std::string graph_index_file = index_prefix + "_disk.index.dgai";
  if (!file_exists(graph_index_file))
    graph_index_file = index_file;

  std::cout << "Reading graph from " << graph_index_file << " (location order)" << std::endl;

  uint64_t vec_data_len = meta.data_dim * sizeof(T);
  uint64_t max_node_len = meta.max_node_len;
  uint64_t nnodes_per_sector = meta.nnodes_per_sector;
  uint64_t range = meta.range;
  uint64_t npoints = meta.npoints;
  uint64_t label_size = meta.label_size;

  uint64_t max_graph_node_len = sizeof(uint32_t) + range * sizeof(uint32_t) + label_size;
  uint64_t nnodes_per_sector_graph = SECTOR_LEN / max_graph_node_len;
  uint64_t bytes_per_write_graph = nnodes_per_sector_graph > 0 ? SECTOR_LEN : ROUND_UP(max_graph_node_len, SECTOR_LEN);
  uint64_t nodes_per_write_graph = nnodes_per_sector_graph > 0 ? nnodes_per_sector_graph : 1;

  uint64_t vec_stride = vec_data_len;
  uint64_t nnodes_per_sector_vec = SECTOR_LEN / vec_stride;
  uint64_t bytes_per_write_vec = nnodes_per_sector_vec > 0 ? SECTOR_LEN : ROUND_UP(vec_stride, SECTOR_LEN);
  uint64_t nodes_per_write_vec = nnodes_per_sector_vec > 0 ? nnodes_per_sector_vec : 1;

  std::string part_file = index_prefix + "_partition.bin.dgai";
  uint64_t total_loc = npoints;
  if (file_exists(part_file)) {
    std::ifstream pin(part_file, std::ios::binary);
    uint64_t C, partition_nums, nd;
    pin.read((char *) &C, sizeof(uint64_t));
    pin.read((char *) &partition_nums, sizeof(uint64_t));
    pin.read((char *) &nd, sizeof(uint64_t));
    total_loc = partition_nums * C;
    pin.close();
    std::cout << "Partition file found: C=" << C << " nparts=" << partition_nums
              << " total_loc=" << total_loc << " (npoints=" << npoints << ")" << std::endl;
  }

  uint64_t graph_padded_bytes = DIV_ROUND_UP(total_loc, nodes_per_write_graph) * SECTOR_LEN;
  uint64_t vec_padded_bytes = DIV_ROUND_UP(total_loc, nodes_per_write_vec) * SECTOR_LEN;

  int graph_in_fd = open(graph_index_file.c_str(), O_RDONLY);
  if (graph_in_fd < 0) {
    std::cerr << "Error: cannot open " << graph_index_file << std::endl;
    return -1;
  }
  int vec_in_fd = open(index_file.c_str(), O_RDONLY);
  if (vec_in_fd < 0) {
    std::cerr << "Error: cannot open " << index_file << std::endl;
    close(graph_in_fd);
    return -1;
  }
  int graph_out_fd = open(graph_file.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (graph_out_fd < 0) {
    std::cerr << "Error: cannot create " << graph_file << std::endl;
    close(graph_in_fd);
    close(vec_in_fd);
    return -1;
  }
  int vec_out_fd = open(vector_file.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (vec_out_fd < 0) {
    std::cerr << "Error: cannot create " << vector_file << std::endl;
    close(graph_in_fd);
    close(vec_in_fd);
    close(graph_out_fd);
    return -1;
  }
  if (ftruncate(graph_out_fd, (off_t) graph_padded_bytes) != 0 || ftruncate(vec_out_fd, (off_t) vec_padded_bytes) != 0) {
    std::cerr << "Error: ftruncate failed for output files" << std::endl;
    close(graph_in_fd);
    close(vec_in_fd);
    close(graph_out_fd);
    close(vec_out_fd);
    return -1;
  }

  std::cout << "Reading vectors from " << index_file << " (id order)" << std::endl;

  int rc = extract_graph_pass<T>(graph_in_fd, graph_out_fd, meta, total_loc, max_graph_node_len, bytes_per_write_graph,
                                 nodes_per_write_graph, range, label_size);
  if (rc == 0)
    rc = extract_vector_pass<T>(vec_in_fd, vec_out_fd, meta, npoints, vec_data_len, bytes_per_write_vec,
                                nodes_per_write_vec);

  close(graph_in_fd);
  close(vec_in_fd);
  close(graph_out_fd);
  close(vec_out_fd);

  if (rc != 0)
    return rc;

  {
    std::string meta_file = index_prefix + "_disk.index.meta";
    std::ifstream fin_meta(index_file, std::ios::binary);
    std::vector<char> meta_buf(SECTOR_LEN);
    fin_meta.read(meta_buf.data(), SECTOR_LEN);
    fin_meta.close();
    std::ofstream fmeta(meta_file, std::ios::binary);
    fmeta.write(meta_buf.data(), SECTOR_LEN);
    fmeta.close();
  }

  std::cout << "=== Split done ===" << std::endl;
  std::cout << "Metadata: " << index_prefix << "_disk.index.meta (" << SECTOR_LEN << " bytes)" << std::endl;
  std::cout << "Graph:    " << graph_file << " (" << graph_padded_bytes << " bytes padded)" << std::endl;
  std::cout << "Vectors:  " << vector_file << " (" << vec_padded_bytes << " bytes padded)" << std::endl;
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0]
              << " <index_type (float/int8/uint8)> <index_prefix_path>"
              << std::endl;
    return -1;
  }

  std::string index_type(argv[1]);
  std::string index_prefix(argv[2]);

  int ret = -1;
  if (index_type == "float")
    ret = split_disk_index<float>(index_prefix);
  else if (index_type == "int8")
    ret = split_disk_index<int8_t>(index_prefix);
  else if (index_type == "uint8")
    ret = split_disk_index<uint8_t>(index_prefix);
  else
    std::cout << "Unsupported type. Use float, int8 or uint8." << std::endl;

  return ret;
}
