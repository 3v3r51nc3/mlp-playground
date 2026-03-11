#include "../include/vector_utils.h"
#include <iostream>
#include <numeric>
#include <iomanip>

void VectorUtils::print(const std::vector<int>& vec, const std::string& name) {
    std::cout << name << " (" << vec.size() << "): { ";
    for (const auto& val : vec) {
        std::cout << std::fixed << std::setprecision(4) << val << " ";
    }
    std::cout << "}\n";
}

void VectorUtils::print(const std::vector<bool>& vec, const std::string& name) {
    std::cout << name << " (" << vec.size() << "): { ";
    for (const auto& val : vec) {
        std::cout << std::fixed << std::setprecision(4) << val << " ";
    }
    std::cout << "}\n";
}

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
        std::cerr << "Error: vectors have different sizes!" << '\n';
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

std::vector<double> VectorUtils::vec_mat_mul(const std::vector<double>& vec, const Matrix& mat) {
    if (vec.size() != static_cast<size_t>(mat.rows)) {
        throw std::invalid_argument("vector size must be equal to matrix rows");
    }

    std::vector<double> result(mat.cols, 0.0);

    for (int col = 0; col < mat.cols; ++col) {
        double sum = 0.0;
        for (int row = 0; row < mat.rows; ++row) {
            sum += vec[row] * mat(row, col);
        }
        result[col] = sum;
    }

    return result;
}

std::vector<double> VectorUtils::elementwise_multiply(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("vectors must have the same size for elementwise multiplication");
    }
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b[i];
    }
    return result;
}


void VectorUtils::add_bias(std::vector<double>& vec, const std::vector<double>& bias) {
    if (vec.size() != bias.size()) {
        throw std::invalid_argument("vector and bias size must match");
    }
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] += bias[i];
    }
}

static double activate(double x, ActivationType activation_type){
    switch (activation_type) {
    case ActivationType::sigmoid:
        return Activation::sigmoid(x);
    case ActivationType::tanh:
        return Activation::tanh(x);
    case ActivationType::relu:
        return Activation::relu(x);
    case ActivationType::leaky_relu:
        return Activation::leaky_relu(x);
    case ActivationType::linear:
    default:
        return Activation::linear(x);
    }
}

static double activate_derivative(double y, ActivationType activation_type) {
    switch (activation_type) {
    case ActivationType::sigmoid:
        return Activation::sigmoid_derivative(y);
    case ActivationType::tanh:
        return Activation::tanh_derivative(y);
    case ActivationType::relu:
        return Activation::relu_derivative(y);
    case ActivationType::leaky_relu:
        return Activation::leaky_relu_derivative(y);
    case ActivationType::linear:
    default:
        return Activation::linear_derivative(y);
    }
}

std::vector<double> VectorUtils::apply_activation(const std::vector<double>& vec, ActivationType type) {
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = activate(vec[i], type);
    }
    return result;
}

std::vector<double> VectorUtils::apply_activation_derivative(const std::vector<double>& vec, ActivationType type)
{
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = activate_derivative(vec[i], type);
    }
    return result;
}


void VectorUtils::add_inplace(std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("size mismatch in add_inplace");
    for (size_t i = 0; i < a.size(); ++i) a[i] += b[i];
}

std::vector<double> VectorUtils::scalar_multiply(const std::vector<double>& vec, double scalar) {
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) result[i] = vec[i] * scalar;
    return result;
}

std::vector<double> VectorUtils::mat_vec_mul(const Matrix& mat, const std::vector<double>& vec) {
    if (static_cast<size_t>(mat.cols) != vec.size()) throw std::invalid_argument("size mismatch in mat_vec_mul");
    std::vector<double> result(mat.rows, 0.0);
    for (int row = 0; row < mat.rows; ++row) {
        double sum = 0;
        for (int col = 0; col < mat.cols; ++col) {
            sum += mat(row, col) * vec[col];
        }
        result[row] = sum;
    }
    return result;
}

Matrix VectorUtils::outer_product(const std::vector<double>& a, const std::vector<double>& b) {
    Matrix result(static_cast<int>(a.size()), static_cast<int>(b.size()));
    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j) {
            result[i][j] = a[i] * b[j];
        }
    }
    return result;
}
