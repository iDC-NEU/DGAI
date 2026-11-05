#include <cstring>
#include <iostream>
#include <string>

#include "utils/index_build_utils.h"

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout << "Correct usage: " << argv[0] << " <type[uint8/float/int8]> <file>" << std::endl;
    exit(-1);
  }

  char *type = argv[1];
  char *base_data_file = argv[2];
  std::string out_file = std::string(base_data_file) + "_normalized";

  if (strcmp(type, "uint8") == 0) {
    pipeann::normalize_data_file<uint8_t>(base_data_file, out_file);
  } else if (strcmp(type, "int8") == 0) {
    pipeann::normalize_data_file<int8_t>(base_data_file, out_file);
  } else if (strcmp(type, "float") == 0) {
    pipeann::normalize_data_file<float>(base_data_file, out_file);
  } else {
    std::cout << "Unknown type: " << type << std::endl;
    exit(-1);
  }
}
