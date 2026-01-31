#include "../include/mnist_loader.h"
#include <fstream>
#include <iostream>
#include <cstdlib>

uint32_t MnistLoader::swap_endian(uint32_t val) {
#ifdef _MSC_VER
    return _byteswap_ulong(val);
#else
    return __builtin_bswap32(val); // GCC/Clang intrinsic
#endif
}

std::vector<std::vector<uint8_t>> MnistLoader::load_images(const std::string& path, int& num_images, int& rows, int& cols) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << '\n';
        std::exit(1);
    }

    uint32_t magic_number = 0;
    file.read((char*)&magic_number, 4);
    magic_number = swap_endian(magic_number);

    file.read((char*)&num_images, 4);
    num_images = swap_endian(num_images);

    file.read((char*)&rows, 4);
    rows = swap_endian(rows);

    file.read((char*)&cols, 4);
    cols = swap_endian(cols);

    std::vector<std::vector<uint8_t>> images(num_images, std::vector<uint8_t>(rows * cols));
    for (int i = 0; i < num_images; ++i) {
        file.read((char*)images[i].data(), rows * cols);
    }

    return images;
}

std::vector<uint8_t> MnistLoader::load_labels(const std::string& path, int& num_labels) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << '\n';
        std::exit(1);
    }

    uint32_t magic_number = 0;
    file.read((char*)&magic_number, 4);
    magic_number = swap_endian(magic_number);

    file.read((char*)&num_labels, 4);
    num_labels = swap_endian(num_labels);

    std::vector<uint8_t> labels(num_labels);
    file.read((char*)labels.data(), num_labels);

    return labels;
}
