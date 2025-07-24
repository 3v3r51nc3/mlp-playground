#include "../include/neural_network.h";
#include "../include/vector_utils.h"
#include <iostream>
#include <iomanip>

NeuralNetwork::NeuralNetwork(Matrix input, Matrix target, double learning_rate, ActivationType activation, double mse_stop_point) : inputs(input), 
targets(target), learning_rate(learning_rate), activation_type(activation), mse_stop_point(mse_stop_point) {

}

NeuralNetwork::~NeuralNetwork() {

}

void NeuralNetwork::train(int epoch_times) {
	std::cout << "training started!\n\n";

	int print_every_n_epochs = 1;
	if (epoch_times > 10000) print_every_n_epochs = 1000;
	else if (epoch_times > 1000) print_every_n_epochs = 100;
	else if (epoch_times > 100) print_every_n_epochs = 10;

	for (int epoch = 0; epoch < epoch_times; epoch++) {
		double epoch_error_sum = 0.0;

		int sample_count = inputs.rows;
		for (int i = 0; i < sample_count; i++) {
	
				// ----- forward pass -----
				std::vector<double> input = inputs.getRow(i);
				std::vector<std::vector<double>> activations; // для хранения выходов всех слоёв
				activations.push_back(input);

				// forward через всю сеть
				for (auto& layer : layers) {
					input = layer.forward(input);
					activations.push_back(input);
				}
				std::vector<double> prediction = input; // выход последнего слоя

				// ----- error calculation -----
				std::vector<double> deltas(targets.cols, 0.0);
				double sample_mse = 0.0;

				for (int d = 0; d < deltas.size(); d++) {
					double delta = targets.at(i, d) - prediction[d];
					deltas[d] = delta;
					sample_mse += delta * delta;
				}
				sample_mse /= deltas.size();
				epoch_error_sum += sample_mse;

				// ----- backward pass -----
				for (int l = layers.size() - 1; l >= 0; l--) {
					std::vector<double> input_to_layer = activations[l]; // входы в текущий слой
					deltas = layers[l].backward(input_to_layer, deltas, learning_rate);

					// VectorUtils::print(deltas, "deltas from backpropagation");
					// (пока пропускаем вычисление новых дельт — добавишь позже при backprop через всю сеть)

				}
				//layer.backward(input, deltas, learning_rate);
			
			// optional: print per-sample MSE
			//std::cout << "sample " << i + 1 << ": mse = " << sample_mse << "\n";
		}

		double epoch_mse = epoch_error_sum / sample_count;
		if ((epoch + 1) % print_every_n_epochs == 0 || epoch == 0 || epoch_mse < 0.0001) {
			std::cout << "epoch " << epoch + 1 << " | mse = " << epoch_mse << "\n";
		}
		if (epoch_mse < mse_stop_point) {
			std::cout << std::fixed << std::setprecision(4);
			std::cout << "MSE is too low ("<< epoch_mse << "), no need to continue training... Quitting cycle"  << "\n";
			std::cout.unsetf(std::ios::fixed); // чтобы вернуть в обычный режим вывода
			std::cout << std::setprecision(2); // можно сбросить, если нужно
			break;
		}
	}

	std::cout << "\ntraining complete!\n";
}


std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
	_ASSERT(input.size() == inputs.cols);
	std::vector<double> output = input;

	if (debug_info) VectorUtils::print(output, "input");
	for (auto& layer : layers) {
		output = layer.forward(output);
	}

	if (debug_info) VectorUtils::print(output, "prediction");

	return output;
}

Layer& NeuralNetwork::add_layer(int input_size, int output_size, bool is_output) {
	Layer new_layer(input_size, output_size, is_output);
	layers.push_back(new_layer);

	return layers[layers.size() - 1];
}

void NeuralNetwork::insert_layer(int neurons_count) {
	if (layers.empty()) {
		layers.push_back(Layer(inputs.cols, neurons_count));
		layers.push_back(Layer(neurons_count, targets.cols));
	}
	else {
		// сохраним выходной слой
		Layer output_layer = std::move(layers.back());
		int output_output_size = output_layer.getOutputSize();

		// удаляем выходной слой
		layers.pop_back();

		// вычисляем размер входа для нового скрытого слоя
		int prev_output_size;
		if (layers.empty()) {
			// если после pop_back сеть пуста (значит был 1 слой), вход берём с inputs
			prev_output_size = inputs.cols;
		}
		else {
			// иначе берём выход предпоследнего слоя
			prev_output_size = layers.back().getOutputSize();
		}

		// добавляем новый скрытый слой
		layers.push_back(Layer(prev_output_size, neurons_count));

		// возвращаем выходной слой с подгонкой размеров весов
		layers.push_back(std::move(output_layer));
		layers.back().resize_weights(neurons_count, output_output_size);
	}
}
