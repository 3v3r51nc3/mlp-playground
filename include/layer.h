#pragma once
#include <vector>
#include "matrix.h"

class Layer {
private:
	Matrix weights;
	std::vector<double> last_output;
	std::vector<double> last_pre_activation;
	bool is_output_layer = false; // ← новый флаг
public:
	Layer() {};
	Layer(int input_count, int output_count, bool output = false);

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