#pragma once
#include <vector>
#include <string>

class VectorUtils {
public:
    static void print(const std::vector<double>& vec, const std::string& name = "unknown vector");
    static double mean(const std::vector<double>& data);
    static double dot(const std::vector<double>& a, const std::vector<double>& b);
    static int size(const std::vector<double>& data);
};
