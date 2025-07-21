#include "../include/layer.h"
#include "../include/vector_utils.h"
#include <iostream>

Layer::Layer(int input_count, int output_count, bool output, ActivationType activation) {
	activation_type = activation;

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
	VectorUtils::print(biases, "biases");

	weights.print("Weights initial values: ");
	is_output_layer = output;
}

double Layer::activate(double x) const {
	switch (activation_type) {
	case ActivationType::sigmoid:
		return 1.0 / (1.0 + std::exp(-x));
	case ActivationType::relu:
		return x > 0.0 ? x : 0.0;
	case ActivationType::leaky_relu:
		return x > 0.0 ? x : 0.01 * x;
	case ActivationType::linear:
	default:
		return x;
	}
}

double Layer::activate_derivative(double y) const {
	switch (activation_type) {
	case ActivationType::sigmoid:
		return y * (1.0 - y);
	case ActivationType::relu:
		return y > 0.0 ? 1.0 : 0.0;
	case ActivationType::leaky_relu:
		return y > 0.0 ? 1.0 : 0.01;
	case ActivationType::linear:
	default:
		return 1.0;
	}
}


//АКТИВАЦИЯ ТОЛЬКО МЕЖДУ СЛОЯМИ НЕ НА ВЫХОДЕ
std::vector<double> Layer::forward(const std::vector<double>& data) {
	std::vector<double> result(weights.cols, 0);
	last_pre_activation.resize(weights.cols);  // выделяем память

	for (int i = 0; i < weights.cols; i++) {
		double sum = 0;
		for (int j = 0; j < weights.rows; j++) {
			sum += data[j] * weights.at(j, i);
		}
		sum += biases[i];
		last_pre_activation[i] = sum;
		
		//result[i] = is_output_layer ? sum : activate(sum);
		if (is_output_layer) {
			if (activation_type == ActivationType::linear) {
				result[i] = sum;
			}
			else {
				result[i] = activate(sum);
			}
		}
		else {
			result[i] = activate(sum);
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
	_ASSERT(deltas.size() == weights.cols);
	_ASSERT(input.size() == weights.rows);

	std::vector<double> activated_deltas(deltas.size());
	for (size_t i = 0; i < deltas.size(); i++) {
		if (activation_type == ActivationType::sigmoid) {
			activated_deltas[i] = deltas[i] * activate_derivative(last_output[i]);
		}
		else { // relu/leaky relu
			activated_deltas[i] = deltas[i] * activate_derivative(last_pre_activation[i]);
		}

	}

	// сохраняем текущие веса в отдельную матрицу перед обновлением,
	// потому что для вычисления новых дельт (ошибок для предыдущего слоя)
	// нужно использовать именно веса, которые были во время прямого прохода (forward pass),
	// а не уже обновленные.
	// это важно, чтобы не исказить градиенты и сохранить корректное направление обучения

	Matrix old_weights = weights;

	for (int d = 0; d < deltas.size(); d++) {
		double bias_correction = learning_rate * activated_deltas[d];
		biases[d] += bias_correction;
		for (int w = 0; w < weights.rows; w++) {
			double correction = learning_rate * activated_deltas[d] * input[w];
			weights.addValue(w, d, correction);
		}
	}

	/*for (int w = 0; w < weights.rows; w++) {
		for (int d = 0; d < deltas.size(); d++) {
			double correction = learning_rate * activated_deltas[d] * input[w];
			
			weights.addValue(w, d, correction);
		}
	}*/

	// рассчитываем новые дельты для предыдущего слоя,
	// которые передадим дальше назад по цепочке слоев.
	// для этого перемножаем старые веса на дельты текущего слоя.
	// используем старые веса, потому что они соответствуют моменту прямого прохода,
	// и отражают реальное влияние входов на текущий слой.
	// вычисление новых дельт:
	// new_delta[i] = sum_j (old_weights[i][j] * deltas[j])

	std::vector<double> new_deltas(weights.rows, 0.0);
	for (int row = 0; row < old_weights.rows; row++) {
		for (int col = 0; col < old_weights.cols; col++) {
			new_deltas[row] += old_weights.at(row, col) * activated_deltas[col];//deltas[col];
		}
	}
	
	return new_deltas;
}


void Layer::resize_weights(int input_count, int output_count) {
	weights.resize(input_count, output_count);
}

void Layer::print(const std::string& name) const {
	std::cout << name << " weights (" << weights.rows << "x" << weights.cols << "):\n";
	weights.print();  // используем метод print из Matrix
}