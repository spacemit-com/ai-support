#include "../utils/json.hpp"

#include <fstream>
#include <iostream>

using json = nlohmann::json;

int main() {
  std::ifstream f("../../../data/config/yolov6.json");
  json config = json::parse(f);
  if (config.contains("name")) {
    std::string name = config["name"];
    std::cout << name << std::endl;
  }
  return 0;
}