#include <omp.h>
#include <cstring>
#include <ctime>
#include "utils/timer.h"
#include "utils/log.h"

#include "ssd_index.h"
#include "utils.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "Correct usage: " << argv[0] << " <base_partition_file>" << std::endl;
    exit(-1);
  }

  int arg_no = 1;
  char *partition_file = argv[arg_no++];
  std::string out_partition_file = std::string(partition_file) + ".aligned";

  std::ifstream part(partition_file);
  std::ofstream out_part(out_partition_file.data(), std::ios::binary);
  uint64_t C, partition_nums, nd;
  part.read((char *) &C, sizeof(uint64_t));
  part.read((char *) &partition_nums, sizeof(uint64_t));
  part.read((char *) &nd, sizeof(uint64_t));
  out_part.write((char *) &C, sizeof(uint64_t));
  out_part.write((char *) &partition_nums, sizeof(uint64_t));
  out_part.write((char *) &nd, sizeof(uint64_t));

  uint32_t tmp_arr[4096];
  for (unsigned i = 0; i < partition_nums; i++) {
    unsigned s;
    part.read((char *) &s, sizeof(unsigned));
    // LOG(INFO) << "Partition " << i << " size " << s;
    // Normally: partition_nums (200000) * C (5) for SIFT1M
    part.read((char *) tmp_arr, sizeof(unsigned) * s);
    for (uint32_t j = s; j < C; ++j) {
      tmp_arr[j] = pipeann::kInvalidID;
    }
    out_part.write((char *) &s, sizeof(unsigned));
    out_part.write((char *) tmp_arr, sizeof(unsigned) * C);
  }
}