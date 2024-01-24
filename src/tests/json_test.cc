#include "../utils/json.hpp"

#include <fstream>
#include <iostream>

using json = nlohmann::json;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <json_file_path>\n";
    return 0;
  }
  std::ifstream f(argv[1]);
  json config = json::parse(f);
  if (config.contains("name")) {
    std::string name = config["name"];
    std::cout << name << std::endl;
  }
  return 0;
}