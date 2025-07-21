#include <iostream>
#include "include/neural_network.h"
#include "include/matrix.h"
#include "include/vector_utils.h"
int main() {
	//6x4
	/*Matrix inputs = Matrix({ //weather conditions as example (6 rows (samples), 5 cols (params))
			{20.0, 0.40, 3.0, 0.95, 0.05},  // почти идеальные условия — можно
			{18.0, 0.60, 6.5, 0.8, 0.2},   // ветер средний — ограничено
			{16.0, 0.75, 8.0, 0.6, 0.3},   // ветер, влажность — ограничено
			{14.0, 0.85, 10.0, 0.4, 0.6},  // почти буря — нельзя
			{23.0, 0.35, 2.5, 0.98, 0.0},  // супер — можно
			{19.0, 0.70, 7.0, 0.7, 0.4}    // не идеально, но сойдёт — ограничено
		});;*/
	Matrix inputs = Matrix({
			{0.8696, 0.25, 0.1765, 0.9167, 0.0833},
			{0.5217, 0.75, 0.6176, 0.5833, 0.3333},
			{0.3043, 1.0, 0.8235, 0.0833, 0.5},
			{0.0,    1.25, 1.0,    0.0,    1.0},
			{1.0,    0.0,  0.0,    1.0,    0.0},
			{0.6522, 0.875, 0.7059, 0.5,   0.6667}
	});

	inputs.print("Weather conditions");

	//6x3 same as input also (bad example)
	Matrix targets = Matrix({ //drone decisions as example
		{0, 0, 1}, // можно
		{0, 1, 0}, // ограничено
		{0, 1, 0}, // ограничено
		{1, 0, 0}, // нельзя
		{0, 0, 1}, // можно
		{0, 1, 0}  // ограничено
		});
	targets.print("Output target values: ");

	NeuralNetwork neural_network(inputs, targets, 0.1, ActivationType::leaky_relu); //sigmoid = 0.1 //ReLU 0.01 or 0.001
		
	constexpr int NETWORK_MODE = 3;
	if (NETWORK_MODE == 0) {
		// no hidden layers
		neural_network.add_layer(inputs.cols, targets.cols, true);
	}
	else if (NETWORK_MODE == 1) { // relu: вообще не работает (ошибка застряет, learning_rate = 0.01)
		// one hidden layer (6 нейронов)
		neural_network.add_layer(inputs.cols, 6);
		neural_network.add_layer(6, targets.cols, true);
	}
	else if (NETWORK_MODE == 2) { // relu: вообще не работает (ошибка застряет, learning_rate = 0.01)
		// two hidden layers (6 -> 6)
		neural_network.add_layer(inputs.cols, 6);
		neural_network.add_layer(6, 6);
		neural_network.add_layer(6, targets.cols, true);
	}
	else if (NETWORK_MODE == 3) { // relu: вообще не работает (ошибка застряет, learning_rate = 0.01), doesn't work
		// two hidden layers (16 -> 8 -> 4)
		neural_network.add_layer(inputs.cols, 16);
		neural_network.add_layer(16, 8);
		neural_network.add_layer(8, 4);
		neural_network.add_layer(4, targets.cols, true);
	}
	else if (NETWORK_MODE == 4) { // relu: вообще не работает (ошибка застряет, learning_rate = 0.01), sigmoid: плохо работает (learning_rate = 0.1)
		// two hidden layers (4 -> 8 -> 16)
		neural_network.add_layer(inputs.cols, 4);
		neural_network.add_layer(4, 8);
		neural_network.add_layer(8, 16);
		neural_network.add_layer(16, targets.cols, true);
	}

	neural_network.train(10000);

	Matrix test_inputs = Matrix({
			{0.9130, 0.125, 0.1176, 0.9583, 0.0417},  // почти идеально — должно быть {0, 0, 1}
			{0.4783, 0.625, 0.5882, 0.6667, 0.2917},  // средне — {0, 1, 0}
			{0.2174, 1.125, 0.9412, 0.2083, 0.7083},  // почти буря — {1, 0, 0}
			{0.8261, 0.375, 0.2941, 0.875, 0.125},    // хорошее — {0, 0, 1}
			{0.3913, 0.8125, 0.7647, 0.4167, 0.4167}  // граничное — {0, 1, 0}
	});


	std::cout << "Predicting! Testing!\n\n";
	for (int i = 0; i < test_inputs.rows; i++) {
		std::cout << "Prediction " << i + 1 << ": \n";
		neural_network.predict(test_inputs.getRow(i));
		//VectorUtils::print(targets.getRow(i), "targets");
	}
	

	std::cout << "\nHello world!\n";
	system("pause");
	return 0;
}

/*

я устал пойду отдыхать
22:22 20/07/2025

ок, давай разберёмся по шагам, почему сеть тупит и mse висит на 0.33 без изменений.

нормализация — хорошая идея, но она не решит основную проблему. скорее всего, у тебя слишком большой разброс значений, и relu без смещения плохо обучается.

отсутствие смещений (bias) — у тебя в слое только веса, но нет bias, а без них relu сильно ограничена и не может сдвигать функцию активации, из-за чего обучение может застопориться.

вектор дельт в обратном проходе — в backward ты используешь deltas, но умножаешь их на relu_derivative(last_output[i]), а last_output[i] — это уже активация relu (всегда >=0), а производную нужно считать по сумме входов до активации, а не по выходу relu.

веса обновляешь по learning_rate * activated_deltas[d] * input[w], но активация дельт неверна, плюс не учитываешь bias



*/