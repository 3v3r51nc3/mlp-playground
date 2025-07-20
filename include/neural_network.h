#pragma once
#include <vector>
#include "layer.h"
#include "matrix.h"

class NeuralNetwork {
private:
	Matrix inputs;
	std::vector<Layer> layers;
	Matrix targets;
	double learning_rate;

	
public:
	NeuralNetwork() {

		layers = { Layer(inputs.cols, targets.cols) };
		learning_rate = 0.001;
	}

	NeuralNetwork(Matrix inputs, Matrix targets, double learning_rate);
	~NeuralNetwork();

	void train(int epoch_count);
	std::vector<double> predict(const std::vector<double>& input);

	Layer& add_layer(int input_size, int output_size, bool is_output = false);
	void insert_layer(int neurons_count);
};