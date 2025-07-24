#pragma once
#include <vector>
#include <cmath>
#include "matrix.h"

enum class ActivationType {
	sigmoid,
	relu,
	leaky_relu,
	linear
};

class Layer {
private:
	Matrix weights;
	std::vector<double> last_output;
	std::vector<double> last_pre_activation;
	std::vector<double> biases;
	bool is_output_layer = false;
	ActivationType activation_type = ActivationType::sigmoid;

	double activate(double x) const;
	double activate_derivative(double activated_output) const;

	bool debug_info;
public:
	Layer() {};
	Layer(int input_count, int output_count, bool output = false, ActivationType activation = ActivationType::sigmoid, bool debug_info = false);

	std::vector<double> forward(const std::vector<double>& data);
	std::vector<double> backward(const std::vector<double>& input,
		const std::vector<double>& deltas,
		double learning_rate);

	void resize_weights(int input_count, int output_count);

	int getInputSize() const {
		return weights.rows;  // число входов
	}

	int getOutputSize() const {
		return weights.cols;  // число выходов
	}

	void print(const std::string& name = "Layer") const;
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

	inline double linear_derivative(double y) {
		return 1.0;
	}
}
