// Reorder a combined DiskANN index file using a partition.bin.aligned mapping.
// Usage: reorder_index <data_type> <index_prefix>
//   - Reads <index_prefix>_disk.index and <index_prefix>_partition.bin.dgai
//   - Writes <index_prefix>_disk.index.dgai
//
// Reorder modes (env REORDER_MODE):
//   auto   - load all nodes into RAM if MemAvailable is sufficient, else mmap scatter
//   memory - force in-memory reorder + sequential write
//   mmap   - sequential input read + mmap scatter write

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <new>
#include <string>
#include <vector>

#include "utils.h"
#include "ssd_index_defs.h"

namespace {

enum class ReorderMode { Auto, Memory, Mmap };

uint64_t read_uint64(std::ifstream &in) {
  uint64_t v;
  in.read((char *) &v, sizeof(uint64_t));
  return v;
}

uint32_t read_uint32(std::ifstream &in) {
  uint32_t v;
  in.read((char *) &v, sizeof(uint32_t));
  return v;
}

uint64_t get_mem_available_bytes() {
  std::ifstream meminfo("/proc/meminfo");
  if (!meminfo.is_open())
    return 0;
  std::string key;
  uint64_t kb = 0;
  while (meminfo >> key) {
    if (key == "MemAvailable:") {
      meminfo >> kb;
      return kb * 1024ULL;
    }
    meminfo.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return 0;
}

ReorderMode parse_reorder_mode() {
  const char *env = std::getenv("REORDER_MODE");
  if (env == nullptr)
    return ReorderMode::Auto;
  if (std::strcmp(env, "memory") == 0)
    return ReorderMode::Memory;
  if (std::strcmp(env, "mmap") == 0)
    return ReorderMode::Mmap;
  return ReorderMode::Auto;
}

void build_loc2id(const std::vector<uint32_t> &id2loc, uint64_t npoints, uint64_t total_loc,
                  std::vector<uint32_t> &loc2id) {
  loc2id.assign(total_loc, pipeann::kInvalidID);
  for (uint32_t id = 0; id < npoints; id++) {
    uint32_t out_loc = id2loc[id];
    if (out_loc != pipeann::kInvalidID)
      loc2id[out_loc] = id;
  }
}

template<typename T>
int load_input_nodes(int in_fd, uint64_t npoints, uint64_t max_node_len, uint64_t nnodes_per_sector,
                     std::vector<char> &in_nodes) {
  const uint64_t bytes_per_io = nnodes_per_sector > 0 ? SECTOR_LEN : ROUND_UP(max_node_len, SECTOR_LEN);
  const uint64_t nodes_per_io = nnodes_per_sector > 0 ? nnodes_per_sector : 1;
  const uint64_t input_data_sectors = nnodes_per_sector > 0 ? DIV_ROUND_UP(npoints, nnodes_per_sector) : npoints;

  in_nodes.resize(npoints * max_node_len);
  std::atomic<uint64_t> sectors_done{0};
  const uint64_t report_every_sectors = std::max<uint64_t>(1, 500000 / nodes_per_io);
  int rc = 0;

  std::cout << "Loading input nodes into memory (" << (in_nodes.size() / (1024 * 1024)) << " MiB, "
            << input_data_sectors << " sectors)" << std::endl;

#pragma omp parallel
  {
    std::vector<char> in_buf(bytes_per_io);
#pragma omp for schedule(dynamic, 256)
    for (int64_t in_sec = 0; in_sec < (int64_t) input_data_sectors; in_sec++) {
      off_t in_off = (off_t) SECTOR_LEN + (uint64_t) in_sec * bytes_per_io;
      if (pread(in_fd, in_buf.data(), bytes_per_io, in_off) != (ssize_t) bytes_per_io) {
#pragma omp critical
        {
          std::cerr << "Error: pread failed at sector " << in_sec << std::endl;
          rc = -1;
        }
        continue;
      }

      uint64_t base_id = (uint64_t) in_sec * nodes_per_io;
      for (uint64_t j = 0; j < nodes_per_io; j++) {
        uint64_t id = base_id + j;
        if (id >= npoints)
          break;
        std::memcpy(in_nodes.data() + id * max_node_len, in_buf.data() + j * max_node_len, max_node_len);
      }

      uint64_t sd = sectors_done.fetch_add(1, std::memory_order_relaxed) + 1;
      if (sd % report_every_sectors == 0 || sd == input_data_sectors) {
        uint64_t done = std::min(sd * nodes_per_io, npoints);
#pragma omp critical
        std::cout << "Loaded: " << done << "/" << npoints << std::endl;
      }
    }
  }
  return rc;
}

template<typename T>
int reorder_with_memory(int out_fd, const std::vector<char> &in_nodes, const std::vector<uint32_t> &loc2id,
                        uint64_t npoints, uint64_t total_loc, uint64_t max_node_len, uint64_t nnodes_per_sector,
                        const std::vector<char> &header) {
  const uint64_t bytes_per_io = nnodes_per_sector > 0 ? SECTOR_LEN : ROUND_UP(max_node_len, SECTOR_LEN);
  const uint64_t nodes_per_io = nnodes_per_sector > 0 ? nnodes_per_sector : 1;
  const uint64_t output_data_sectors = nnodes_per_sector > 0 ? DIV_ROUND_UP(total_loc, nnodes_per_sector) : total_loc;
  const uint64_t report_every_sectors = std::max<uint64_t>(1, 500000 / nodes_per_io);

  if (pwrite(out_fd, header.data(), SECTOR_LEN, 0) != (ssize_t) SECTOR_LEN) {
    std::cerr << "Error: failed to write index header" << std::endl;
    return -1;
  }

  std::cout << "Writing reordered index sequentially from memory (" << output_data_sectors << " sectors)"
            << std::endl;

  std::vector<char> out_buf(bytes_per_io, 0);
  for (uint64_t out_sec = 0; out_sec < output_data_sectors; out_sec++) {
    std::fill(out_buf.begin(), out_buf.end(), 0);
    uint64_t base_loc = out_sec * nodes_per_io;

    for (uint64_t j = 0; j < nodes_per_io; j++) {
      uint64_t out_loc = base_loc + j;
      if (out_loc >= total_loc)
        break;
      uint32_t id = loc2id[out_loc];
      if (id == pipeann::kInvalidID)
        continue;
      std::memcpy(out_buf.data() + j * max_node_len, in_nodes.data() + (uint64_t) id * max_node_len, max_node_len);
    }

    off_t out_off = (off_t) SECTOR_LEN + out_sec * bytes_per_io;
    if (pwrite(out_fd, out_buf.data(), bytes_per_io, out_off) != (ssize_t) bytes_per_io) {
      std::cerr << "Error: pwrite failed at output sector " << out_sec << std::endl;
      return -1;
    }

    uint64_t done = std::min(base_loc + nodes_per_io, total_loc);
    if (done % 500000 == 0 || out_sec + 1 == output_data_sectors)
      std::cout << "Written: " << done << "/" << total_loc << std::endl;
  }
  (void) npoints;
  return 0;
}

template<typename T>
int reorder_with_mmap(int in_fd, int out_fd, char *out_map, uint64_t out_file_size,
                      const std::vector<uint32_t> &id2loc, uint64_t npoints, uint64_t max_node_len,
                      uint64_t nnodes_per_sector) {
  const uint64_t bytes_per_io = nnodes_per_sector > 0 ? SECTOR_LEN : ROUND_UP(max_node_len, SECTOR_LEN);
  const uint64_t nodes_per_io = nnodes_per_sector > 0 ? nnodes_per_sector : 1;
  const uint64_t input_data_sectors = nnodes_per_sector > 0 ? DIV_ROUND_UP(npoints, nnodes_per_sector) : npoints;
  const uint64_t report_every_sectors = std::max<uint64_t>(1, 500000 / nodes_per_io);

  madvise(out_map, out_file_size, MADV_RANDOM);
  if (pread(in_fd, out_map, SECTOR_LEN, 0) != (ssize_t) SECTOR_LEN) {
    std::cerr << "Error: failed to read index header" << std::endl;
    return -1;
  }

  std::atomic<uint64_t> sectors_done{0};
  std::cout << "Reordering with sequential input read + mmap scatter (" << input_data_sectors
            << " input sectors)" << std::endl;

  int rc = 0;
#pragma omp parallel
  {
    std::vector<char> in_buf(bytes_per_io);
#pragma omp for schedule(dynamic, 256)
    for (int64_t in_sec = 0; in_sec < (int64_t) input_data_sectors; in_sec++) {
      off_t in_off = (off_t) SECTOR_LEN + (uint64_t) in_sec * bytes_per_io;
      if (pread(in_fd, in_buf.data(), bytes_per_io, in_off) != (ssize_t) bytes_per_io) {
#pragma omp critical
        {
          std::cerr << "Error: pread failed at sector " << in_sec << std::endl;
          rc = -1;
        }
        continue;
      }

      uint64_t base_id = (uint64_t) in_sec * nodes_per_io;
      for (uint64_t j = 0; j < nodes_per_io; j++) {
        uint64_t id = base_id + j;
        if (id >= npoints)
          break;
        uint32_t out_loc = id2loc[id];
        if (out_loc == pipeann::kInvalidID)
          continue;
        uint64_t out_sec = out_loc / nodes_per_io;
        uint64_t slot = out_loc % nodes_per_io;
        char *dst = out_map + SECTOR_LEN + out_sec * bytes_per_io + slot * max_node_len;
        std::memcpy(dst, in_buf.data() + j * max_node_len, max_node_len);
      }

      uint64_t sd = sectors_done.fetch_add(1, std::memory_order_relaxed) + 1;
      if (sd % report_every_sectors == 0 || sd == input_data_sectors) {
        uint64_t done = std::min(sd * nodes_per_io, npoints);
#pragma omp critical
        std::cout << "Rewritten: " << done << "/" << npoints << std::endl;
      }
    }
  }

  if (rc != 0)
    return rc;
  if (msync(out_map, out_file_size, MS_SYNC) != 0)
    std::cerr << "Warning: msync failed" << std::endl;
  return 0;
}

}  // namespace

template<typename T>
int reorder_index_template(const std::string &index_prefix) {
  std::string index_file = index_prefix + "_disk.index";
  std::string part_file = index_prefix + "_partition.bin.dgai";
  std::string out_file = index_prefix + "_disk.index.dgai";

  pipeann::SSDIndexMetadata<T> meta;
  meta.load_from_disk_index(index_file);
  meta.print();
  uint64_t npoints = meta.npoints;
  uint64_t max_node_len = meta.max_node_len;
  uint64_t nnodes_per_sector = meta.nnodes_per_sector;
  uint64_t bytes_per_io = nnodes_per_sector > 0 ? SECTOR_LEN : ROUND_UP(max_node_len, SECTOR_LEN);
  uint64_t nodes_per_io = nnodes_per_sector > 0 ? nnodes_per_sector : 1;

  std::ifstream pin(part_file, std::ios::binary);
  if (!pin.is_open()) {
    std::cerr << "Error: cannot open " << part_file << std::endl;
    return -1;
  }
  uint64_t C = read_uint64(pin);
  uint64_t partition_nums = read_uint64(pin);
  uint64_t nd = read_uint64(pin);
  if (nd != npoints) {
    std::cerr << "Error: partition nd=" << nd << " != index npoints=" << npoints << std::endl;
    return -1;
  }
  std::cout << "Partition: C=" << C << " nparts=" << partition_nums << " nd=" << nd << std::endl;

  std::vector<uint32_t> id2loc(npoints, pipeann::kInvalidID);
  uint32_t loc = 0;
  std::vector<uint32_t> slot_buf(C);
  for (uint64_t s = 0; s < partition_nums; s++) {
    (void) read_uint32(pin);
    pin.read((char *) slot_buf.data(), C * sizeof(uint32_t));
    for (uint32_t j = 0; j < C; j++) {
      uint32_t id = slot_buf[j];
      if (id != pipeann::kInvalidID) {
        if (id >= npoints) {
          std::cerr << "Error: id=" << id << " out of range" << std::endl;
          return -1;
        }
        id2loc[id] = loc + j;
      }
    }
    loc += (uint32_t) C;
  }
  uint64_t total_loc = partition_nums * C;
  pin.close();

  uint64_t output_data_sectors = nnodes_per_sector > 0 ? DIV_ROUND_UP(total_loc, nnodes_per_sector) : total_loc;
  uint64_t out_file_size = SECTOR_LEN + output_data_sectors * bytes_per_io;

  int in_fd = open(index_file.c_str(), O_RDONLY);
  if (in_fd < 0) {
    std::cerr << "Error: cannot open " << index_file << std::endl;
    return -1;
  }

  int out_fd = open(out_file.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (out_fd < 0) {
    std::cerr << "Error: cannot create " << out_file << std::endl;
    close(in_fd);
    return -1;
  }
  if (ftruncate(out_fd, (off_t) out_file_size) != 0) {
    std::cerr << "Error: ftruncate failed for " << out_file << std::endl;
    close(in_fd);
    close(out_fd);
    return -1;
  }

  const uint64_t node_bytes = npoints * max_node_len;
  const uint64_t loc2id_bytes = total_loc * sizeof(uint32_t);
  const uint64_t mem_overhead = 1ULL << 30;  // 1 GiB margin
  const uint64_t mem_needed = node_bytes + loc2id_bytes + mem_overhead;
  const uint64_t mem_available = get_mem_available_bytes();
  ReorderMode mode = parse_reorder_mode();
  bool use_memory = mode == ReorderMode::Memory;
  if (mode == ReorderMode::Auto)
    use_memory = mem_available >= mem_needed;

  std::cout << "MemAvailable: " << (mem_available / (1024 * 1024)) << " MiB, need ~"
            << (mem_needed / (1024 * 1024)) << " MiB for in-memory reorder" << std::endl;
  if (mode == ReorderMode::Auto && !use_memory)
    std::cout << "Not enough MemAvailable for in-memory reorder, will use mmap scatter" << std::endl;

  int rc = 0;
  bool memory_done = false;
  if (use_memory) {
    std::cout << "Using in-memory reorder (REORDER_MODE="
              << (mode == ReorderMode::Memory ? "memory" : "auto") << ")" << std::endl;
    std::vector<char> in_nodes;
    std::vector<uint32_t> loc2id;
    std::vector<char> header(SECTOR_LEN);
    try {
      if (pread(in_fd, header.data(), SECTOR_LEN, 0) != (ssize_t) SECTOR_LEN) {
        std::cerr << "Error: failed to read index header" << std::endl;
        close(in_fd);
        close(out_fd);
        return -1;
      }
      rc = load_input_nodes<T>(in_fd, npoints, max_node_len, nnodes_per_sector, in_nodes);
      if (rc == 0) {
        build_loc2id(id2loc, npoints, total_loc, loc2id);
        rc = reorder_with_memory<T>(out_fd, in_nodes, loc2id, npoints, total_loc, max_node_len, nnodes_per_sector,
                                    header);
      }
      memory_done = (rc == 0);
    } catch (const std::bad_alloc &) {
      std::cerr << "Warning: in-memory reorder OOM, falling back to mmap scatter" << std::endl;
      rc = 0;
    }
    if (!memory_done && mode == ReorderMode::Memory) {
      close(in_fd);
      close(out_fd);
      return rc != 0 ? rc : -1;
    }
  }

  if (!memory_done) {
    if (mode == ReorderMode::Memory) {
      std::cerr << "Error: REORDER_MODE=memory but not enough RAM" << std::endl;
      close(in_fd);
      close(out_fd);
      return -1;
    }
    std::cout << "Using mmap scatter reorder (REORDER_MODE="
              << (mode == ReorderMode::Mmap ? "mmap" : "auto") << ")" << std::endl;
    char *out_map = (char *) mmap(nullptr, out_file_size, PROT_READ | PROT_WRITE, MAP_SHARED, out_fd, 0);
    if (out_map == MAP_FAILED) {
      std::cerr << "Error: mmap failed for " << out_file << std::endl;
      close(in_fd);
      close(out_fd);
      return -1;
    }
    rc = reorder_with_mmap<T>(in_fd, out_fd, out_map, out_file_size, id2loc, npoints, max_node_len, nnodes_per_sector);
    munmap(out_map, out_file_size);
  }

  close(in_fd);
  close(out_fd);

  if (rc != 0)
    return rc;
  std::cout << "Reordered index written to " << out_file << std::endl;
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0]
              << " <data_type(float/int8/uint8)> <index_prefix>" << std::endl;
    std::cout << "  Reads <index_prefix>_disk.index and <index_prefix>_partition.bin.dgai" << std::endl;
    std::cout << "  Writes <index_prefix>_disk.index.dgai" << std::endl;
    std::cout << "  Env REORDER_MODE=auto|memory|mmap (default auto)" << std::endl;
    return -1;
  }

  std::string dtype(argv[1]);
  std::string prefix(argv[2]);

  if (dtype == "float")
    return reorder_index_template<float>(prefix);
  else if (dtype == "int8")
    return reorder_index_template<int8_t>(prefix);
  else if (dtype == "uint8")
    return reorder_index_template<uint8_t>(prefix);
  else {
    std::cerr << "Unknown type: " << dtype << std::endl;
    return -1;
  }
}
