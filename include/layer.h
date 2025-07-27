#pragma once
#include <vector>
#include "matrix.h"
#include "../include/vector_utils.h"

class Layer {
private:
	Matrix weights;
	std::vector<double> last_output;
	std::vector<double> last_pre_activation;
	std::vector<double> biases;
	bool is_output_layer = false;
	ActivationType activation_type = ActivationType::sigmoid;

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

	bool isOutputLayer() const {
		return is_output_layer;
	}

	void print(const std::string& name = "Layer") const;
};

