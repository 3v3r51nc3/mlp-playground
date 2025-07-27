#include "../include/layer.h"
#include <iostream>

Layer::Layer(int input_count, int output_count, bool output, ActivationType activation, bool debug_info) {
	activation_type = activation;
	this->debug_info = debug_info;

	double min_val = -1.0, max_val = 1.0;

	switch (activation_type) {
	case ActivationType::sigmoid:
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
	_ASSERT(deltas.size() == weights.cols);
	_ASSERT(input.size() == weights.rows);

	// 1. вычисляем производную активации и активированные дельты: δ̂ = δ ⊙ σ'(z)
	const auto& base_vec = (activation_type == ActivationType::sigmoid) ? last_output : last_pre_activation;
	auto activation_derivative = VectorUtils::apply_activation_derivative(base_vec, activation_type);
	auto activated_deltas = VectorUtils::elementwise_multiply(deltas, activation_derivative);

	// 2. сохраняем копию старых весов для вычисления новых дельт
	Matrix old_weights = weights;

	// 3. обновляем смещения (biases)
	auto scaled_deltas = VectorUtils::scalar_multiply(activated_deltas, learning_rate);
	VectorUtils::add_inplace(biases, scaled_deltas);
	
	// 4. обновляем веса через внешнее произведение и добавляем к матрице весов
	Matrix correction = VectorUtils::outer_product(input, activated_deltas);
	correction *= learning_rate;
	weights += correction; // либо перегрузка оператора +=

	// 5. вычисляем новые дельты для предыдущего слоя: δ^(l-1) = W^T · δ̂
	auto new_deltas = VectorUtils::mat_vec_mul(old_weights, activated_deltas);

	return new_deltas;
}

void Layer::resize_weights(int input_count, int output_count) {
	weights.resize(input_count, output_count);
}

void Layer::print(const std::string& name) const {
	std::cout << name << " weights (" << weights.rows << "x" << weights.cols << "):\n";
	weights.print();  // используем метод print из Matrix
}