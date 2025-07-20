#include "../include/vector_utils.h"
#include <iostream>
#include <numeric>
#include <iomanip>

void VectorUtils::print(const std::vector<double>& vec, const std::string& name) {
    std::cout << name << " (" << vec.size() << "): { ";
    for (const auto& val : vec) {
        std::cout << std::fixed << std::setprecision(4) << val << " ";
    }
    std::cout << "}\n";
}

double VectorUtils::mean(const std::vector<double>& data) {
    if (data.empty()) return 0.0;
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

double VectorUtils::dot(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        std::cerr << "Error: vectors have different sizes!" << std::endl;
        return 0.0;
    }
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

int VectorUtils::size(const std::vector<double>& data) {
    return static_cast<int>(data.size());
}
