#include "../include/layer.h"
#include <iostream>
#include <random>

Layer::Layer(int input_count, int output_count, double dropout_rate, bool output, ActivationType activation, bool debug_info) {
	this->dropout_rate = dropout_rate;
	activation_type = activation;
	this->debug_info = debug_info;

	double min_val = -1.0, max_val = 1.0;

	switch (activation_type) {
	case ActivationType::sigmoid:
	case ActivationType::tanh:
	{
		double limit = sqrt(6.0 / (input_count + output_count));
		min_val = -limit;
		max_val = limit;
		break;
	}
	case ActivationType::relu:
	case ActivationType::leaky_relu: {
		double stddev = sqrt(2.0 / input_count);
		min_val = -stddev;
		max_val = stddev;
		break;
	}
	case ActivationType::linear:
	default:
		break;
	}

	weights = Matrix::random(input_count, output_count, min_val, max_val);

	biases.resize(output_count);
	for (auto& b : biases) {
		b = ((double)rand() / RAND_MAX - 0.5); // от -0.5 до +0.5
	}

	if (debug_info) VectorUtils::print(biases, "biases");
	if (debug_info) weights.print("Weights initial values: ");

	is_output_layer = output;
}

//АКТИВАЦИЯ ТОЛЬКО МЕЖДУ СЛОЯМИ НЕ НА ВЫХОДЕ	(НЕ ВСЕГДА)
std::vector<double> Layer::forward(const std::vector<double>& data) {
	auto z = VectorUtils::vec_mat_mul(data, weights);
	VectorUtils::add_bias(z, biases);
	last_pre_activation = z;

	auto result = (is_output_layer && activation_type == ActivationType::linear) ? z : VectorUtils::apply_activation(z, activation_type);

	// применяем dropout только если включен и это НЕ выходной слой (dropout не для выхода)
	if (dropout_enabled && !is_output_layer) {
		generate_dropout_mask(result.size());

		//VectorUtils::print(dropout_mask, "dropout mask");
		for (int i = 0; i < result.size(); ++i) {
			if (dropout_mask[i]) {
				result[i] /= (1.0 - dropout_rate); // усиливаем активные
			}
			else {
				result[i] = 0.0; // выключаем
			}
		}
	}

	last_output = result;
	return result;
}

// ----- backward pass: weight update -----
// вычисляем корректировку веса (correction) по формуле:
// correction = learning_rate * delta * соответствующее входное значение
//
// суммируем ошибки для подсчёта средней ошибки по всем выходам (MSE)
// после завершения цикла усредняем сумму ошибок на количество выходов,
// чтобы получить среднеквадратичную ошибку для текущего примера
//
// выводим MSE для контроля качества предсказания
//
// дополнительное пояснение по контексту:
//
// - веса хранятся в матрице размером (input.cols, target.cols)
// - каждый вес связан с конкретным входным параметром (строка матрицы)
//   и конкретным выходом (столбец матрицы)
// - при обновлении весов необходимо итерироваться по всем входным параметрам (w)
//   для каждого выхода (p) и обновлять weight[w][p] с использованием правильного входного значения

//weights.rows - вход weights.cols - выход
std::vector<double> Layer::backward(const std::vector<double>& input,
	const std::vector<double>& deltas,
	double learning_rate) {
	auto grads = compute_gradients(input, deltas);

	// обновляем
	grads.dW *= learning_rate;
	weights += grads.dW;
	

	auto scaled_deltas = VectorUtils::scalar_multiply(grads.dB, learning_rate); // grads.dB = activated_deltas i suppose
	VectorUtils::add_inplace(biases, scaled_deltas);

	return grads.next_deltas;
}


// --- методы накопления градиентов ---
LayerGradients Layer::compute_gradients(const std::vector<double>& input,
	const std::vector<double>& deltas) {
	_ASSERT(deltas.size() == weights.cols);
	_ASSERT(input.size() == weights.rows);

	const auto& base_vec = (activation_type == ActivationType::sigmoid)
		? last_output : last_pre_activation;

	auto activation_derivative = VectorUtils::apply_activation_derivative(base_vec, activation_type);
	auto activated_deltas = VectorUtils::elementwise_multiply(deltas, activation_derivative);

	if (dropout_enabled && !is_output_layer) {
		for (int i = 0; i < activated_deltas.size(); ++i) {
			activated_deltas[i] *= dropout_mask[i];
		}
	}

	// dW = input outer activated_deltas
	Matrix dW = VectorUtils::outer_product(input, activated_deltas);

	// dB = activated_deltas
	std::vector<double> dB = activated_deltas;

	// new_deltas = Wᵗ * activated_deltas
	std::vector<double> new_deltas = VectorUtils::mat_vec_mul(weights, activated_deltas);

	return { dW, dB, new_deltas };
}

void Layer::zero_grad_accum() {
	grad_weights_accum = Matrix(weights.rows, weights.cols);
	grad_biases_accum = std::vector<double>(biases.size(), 0.0);
	accum_count = 0;
}

void Layer::accumulate_gradients(const Matrix& dW, const std::vector<double>& dB) {
	for (int i = 0; i < weights.rows; ++i)
		for (int j = 0; j < weights.cols; ++j)
			grad_weights_accum(i, j) += dW(i, j);

	for (int i = 0; i < biases.size(); ++i)
		grad_biases_accum[i] += dB[i];

	accum_count++;
}

void Layer::apply_accumulated_gradients(double learning_rate) {
	for (int i = 0; i < weights.rows; ++i)
		for (int j = 0; j < weights.cols; ++j)
			weights(i, j) += learning_rate * (grad_weights_accum(i, j) / accum_count);

	for (int i = 0; i < biases.size(); ++i)
		biases[i] += learning_rate * (grad_biases_accum[i] / accum_count);

	zero_grad_accum();
}

void Layer::generate_dropout_mask(int size) {
	dropout_mask.resize(size);
	std::random_device rd;
	std::mt19937 gen(rd());
	std::bernoulli_distribution dist(1.0 - dropout_rate);

	for (int i = 0; i < size; ++i) {
		dropout_mask[i] = dist(gen);
	}
}

void Layer::resize_weights(int input_count, int output_count) {
	weights.resize(input_count, output_count);
}

void Layer::print(const std::string& name) const {
	std::cout << name << " weights (" << weights.rows << "x" << weights.cols << "):\n";
	weights.print();  // используем метод print из Matrix
}