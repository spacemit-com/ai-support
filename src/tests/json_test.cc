#include <iostream>
#include <fstream>
#include "../utils/json.hpp"

using json=nlohmann::json;

int main() {
    std::ifstream f("/home/gexy5/Documents/bianbu-support/data/config/yolov6.json");
    json config = json::parse(f);
    std::string name = config["name"];
    int num = config["num"];
    std::vector<int> anchors = config["anchors"];
    std::cout<<name<<std::endl;
    std::cout<<num<<std::endl;
    std::cout<<anchors[0]<<std::endl;
    return 0;
}