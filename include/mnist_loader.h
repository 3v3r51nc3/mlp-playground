#pragma once
#include <vector>
#include <string>
#include <cstdint>

class MnistLoader {
public:
    static uint32_t swap_endian(uint32_t val);

    static std::vector<std::vector<uint8_t>> load_images(const std::string& path, int& num_images, int& rows, int& cols);

    static std::vector<uint8_t> load_labels(const std::string& path, int& num_labels);
};
