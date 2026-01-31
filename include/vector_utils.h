#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#include "../include/matrix.h"

enum class ActivationType {
	softmax,
    sigmoid,
	tanh,
    relu,
    leaky_relu,
    linear
};

namespace Activation {

	inline double sigmoid(double x) {
		return 1.0 / (1.0 + std::exp(-x));
	}

	inline double sigmoid_derivative(double y) {
		return y * (1.0 - y);  // y = sigmoid(x)
	}

	inline double relu(double x) {
		return x > 0.0 ? x : 0.0;
	}

	inline double relu_derivative(double y) {
		return y > 0.0 ? 1.0 : 0.0;
	}

	inline double leaky_relu(double x) {
		return x > 0.0 ? x : 0.01 * x;
	}

	inline double leaky_relu_derivative(double y) {
		return y > 0.0 ? 1.0 : 0.01;
	}

	inline double linear(double x) {
		return x;
	}

	inline double linear_derivative(double ) {
		return 1.0;
	}

	inline double tanh(double x) {
		return std::tanh(x);
	}

	inline double tanh_derivative(double y) {
		return 1.0 - y * y;  // y = tanh(x)
	}

	inline std::vector<double> softmax(const std::vector<double>& x) {
		std::vector<double> result(x.size());
		double max_elem = *std::max_element(x.begin(), x.end()); // для численной стабильности
		double sum = 0.0;
		for (double val : x) {
			sum += std::exp(val - max_elem);
		}
		for (size_t i = 0; i < x.size(); ++i) {
			result[i] = std::exp(x[i] - max_elem) / sum;
		}
		return result;
	}

	inline std::vector<double> softmax_derivative(const std::vector<double>& softmax_output) {
		// упрощённый вариант: возвращает только диагональные элементы Якобиана
		// для полной производной лучше использовать матрицу Якоби (s_i * (δ_ij - s_j))
		std::vector<double> result(softmax_output.size());
		for (size_t i = 0; i < softmax_output.size(); ++i) {
			result[i] = softmax_output[i] * (1.0 - softmax_output[i]);
		}
		return result;
	}

}

class VectorUtils {
public:
	static void print(const std::vector<bool>& vec, const std::string& name = "unknown vector");
	static void print(const std::vector<int>& vec, const std::string& name = "unknown vector");
    static void print(const std::vector<double>& vec, const std::string& name = "unknown vector");
    static double mean(const std::vector<double>& data);
    static double dot(const std::vector<double>& a, const std::vector<double>& b);
    static int size(const std::vector<double>& data);

    // умножение вектора на матрицу (вектор должен быть длины равной числу строк матрицы)
    static std::vector<double> vec_mat_mul(const std::vector<double>& vec, const Matrix& mat);

	static std::vector<double> elementwise_multiply(const std::vector<double>& a, const std::vector<double>& b);

    // прибавление смещения (bias) к вектору
    static void add_bias(std::vector<double>& vec, const std::vector<double>& bias);

    static std::vector<double> apply_activation(const std::vector<double>& vec, ActivationType type);
	static std::vector<double> apply_activation_derivative(const std::vector<double>& vec, ActivationType type);

	// поэлементное сложение (в месте)
	static void add_inplace(std::vector<double>& a, const std::vector<double>& b);

	// умножение вектора на скаляр, возвращает новый вектор
	static std::vector<double> scalar_multiply(const std::vector<double>& vec, double scalar);

	// матрично-векторное умножение
	static std::vector<double> mat_vec_mul(const Matrix& mat, const std::vector<double>& vec);

	// внешнее произведение (outer product), возвращает матрицу
	static Matrix outer_product(const std::vector<double>& a, const std::vector<double>& b);
};
