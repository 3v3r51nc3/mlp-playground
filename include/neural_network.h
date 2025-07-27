#pragma once
#include <vector>
#include "layer.h"
#include "matrix.h"

class NeuralNetwork {
private:
	Matrix inputs;
	std::vector<Layer> layers;
	Matrix targets;

	double dropout_rate;
	double learning_rate;
	double mse_stop_point;

	const bool debug_info = false;
public:
	NeuralNetwork() {
	}

	NeuralNetwork(Matrix inputs, Matrix targets, double learning_rate, double dropout_rate = 0.5, double mse_stop_point = 0.01);
	~NeuralNetwork();

	void train(int epoch_count, GradientDescentType gd_type, int mini_batch_size = 32);
	std::vector<double> predict(const std::vector<double>& input);

	Layer& add_layer(int input_size, int output_size, ActivationType activation, bool is_output = false);
};