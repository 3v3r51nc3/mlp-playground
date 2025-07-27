#include <iostream>
#include "include/neural_network.h"
#include "include/matrix.h"
#include "include/vector_utils.h"
#include "include/mnist_loader.h"

#include <fstream>
#include <cstdint>
#include <filesystem>



namespace fs = std::filesystem;

int main() {

	fs::path current_dir = fs::current_path();

	std::cout << "Current directory: " << current_dir << "\n\n";

	int num_train_images, rows, cols;
	auto train_images = MnistLoader::load_images("datasets/train-images.idx3-ubyte", num_train_images, rows, cols);

	int num_train_labels;
	auto train_labels = MnistLoader::load_labels("datasets/train-labels.idx1-ubyte", num_train_labels);

	//1000 samples
	const int TRAIN_SIZE = 1000;
	Matrix inputs(TRAIN_SIZE, rows * cols);
	Matrix targets(TRAIN_SIZE, 10); // 10 выходов в конце (классов)

	for (int i = 0; i < TRAIN_SIZE; ++i) {
		// проходим по каждому из TRAIN_SIZE обучающих примеров

		for (int j = 0; j < rows * cols; ++j) {
			// для каждого пикселя изображения (rows * cols - размер одного изображения)
			// нормализуем значение пикселя из диапазона [0, 255] в [0, 1] для удобства обучения нейросети
			// 0 - черный, 1 - белый
			inputs(i, j) = train_images[i][j] / 255.0f;
		}

		// инициализируем вектор целей (one-hot encoding) длиной 10 (число классов в MNIST — цифры 0-9)
		for (int j = 0; j < 10; ++j)
			targets(i, j) = 0.0f;

		// ставим 1.0 в позиции, соответствующей метке train_labels[i]
		// таким образом создаём правильный "ответ" для обучения нейросети (one-hot)
		targets(i, train_labels[i]) = 1.0f;
	}

	//training
	NeuralNetwork net(inputs, targets, 0.1f, ActivationType::sigmoid); // можно relu/sigmoid

	net.add_layer(inputs.cols, 64);
	net.add_layer(64, 32);
	net.add_layer(32, 10, true);

	net.train(15); // можно 10–20 эпох — больше не нужно на 1000 примерах


	int num_test_images, test_rows, test_cols;
	auto test_images = MnistLoader::load_images("datasets/t10k-images.idx3-ubyte", num_test_images, test_rows, test_cols);

	int num_test_labels;
	auto test_labels = MnistLoader::load_labels("datasets/t10k-labels.idx1-ubyte", num_test_labels);

	int TEST_SIZE = 100;
	int correct = 0;

	for (int i = 0; i < TEST_SIZE; ++i) {
		std::vector<double> test_input;
		for (int j = 0; j < test_rows * test_cols; ++j)
			test_input.push_back(test_images[i][j] / 255.0);

		auto predicted_vector = net.predict(test_input);

		int predicted = 0;
		double max_value = predicted_vector[0];
		for (int z = 1; z < predicted_vector.size(); z++) {
			if (predicted_vector[z] > max_value) {
				max_value = predicted_vector[z];
				predicted = z;
			}
		}

		int actual = test_labels[i];

		std::cout << "Predicted: " << predicted << " | Actual: " << actual << "\n";

		if (predicted == actual) correct++;
	}

	double accuracy = (double)correct / TEST_SIZE;
	std::cout << "Accuracy on " << TEST_SIZE << " test samples: " << accuracy * 100 << "%\n";
}