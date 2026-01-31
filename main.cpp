#include <iostream>
#include "include/neural_network.h"
#include "include/matrix.h"
#include "include/vector_utils.h"
#include "include/mnist_loader.h"

#include <fstream>
#include <cstdint>
#include <filesystem>
#include <algorithm> // for std::min

namespace fs = std::filesystem;

int main() {

    fs::path current_dir = fs::current_path();
    std::cout << "Current directory: " << current_dir << "\n\n";

    // --- Load Data ---
    int num_train_images, rows, cols;
    auto train_images = MnistLoader::load_images("datasets/train-images.idx3-ubyte", num_train_images, rows, cols);

    int num_train_labels;
    auto train_labels = MnistLoader::load_labels("datasets/train-labels.idx1-ubyte", num_train_labels);

    // SETTINGS: subset size
    // Using 2000 samples makes it fast for CPU but provides better accuracy than 1000.
    const int TRAIN_SIZE = 2000;
    
    // Safety check to ensure we don't exceed available images
    int actual_train_size = std::min(TRAIN_SIZE, num_train_images);
    std::cout << "Preparing training data (" << actual_train_size << " samples)...\n";

    Matrix inputs(actual_train_size, rows * cols);
    Matrix targets(actual_train_size, 10); // 10 output classes (digits 0-9)

    for (int i = 0; i < actual_train_size; ++i) {
        // Iterate through each training example

        for (int j = 0; j < rows * cols; ++j) {
            // Normalize pixel value from [0, 255] to [0, 1]
            // 0 - black, 1 - white
            inputs(i, j) = train_images[i][j] / 255.0f;
        }

        // Initialize target vector (one-hot encoding)
        for (int j = 0; j < 10; ++j)
            targets(i, j) = 0.0f;

        // Set 1.0 at the index corresponding to the label
        targets(i, train_labels[i]) = 1.0f;
    }

    // --- Network Setup ---
    // Learning Rate: 0.1
    NeuralNetwork net(inputs, targets, 0.1, 0); 

    // Define Architecture
    // Input -> Hidden (128) -> Hidden (64) -> Output (10)
    net.add_layer(inputs.cols, 128);
    net.add_layer(128, 64);
    net.add_layer(64, 10, true);

    std::cout << "Starting training...\n";

    // --- Training ---
    // Epochs: 50 (With small data, we need more epochs to converge)
    // Batch Size: 32 (Good balance for small datasets)
    net.train(50, GradientDescentType::Stochastic, 32); 

    // --- Testing ---
    int num_test_images, test_rows, test_cols;
    auto test_images = MnistLoader::load_images("datasets/t10k-images.idx3-ubyte", num_test_images, test_rows, test_cols);

    int num_test_labels;
    auto test_labels = MnistLoader::load_labels("datasets/t10k-labels.idx1-ubyte", num_test_labels);

    // Test on a smaller subset for speed (500 images)
    int TEST_SIZE = 500;
    int actual_test_size = std::min(TEST_SIZE, num_test_images);
    int correct = 0;

    std::cout << "Starting testing on " << actual_test_size << " samples...\n";

    for (int i = 0; i < actual_test_size; ++i) {
        std::vector<double> test_input;
        test_input.reserve(test_rows * test_cols);
        
        for (int j = 0; j < test_rows * test_cols; ++j)
            test_input.push_back(test_images[i][j] / 255.0);

        auto predicted_vector = net.predict(test_input);

        // Find index of the maximum value
        int predicted = 0;
        double max_value = predicted_vector[0];
        for (size_t z = 1; z < predicted_vector.size(); z++) {
            if (predicted_vector[z] > max_value) {
                max_value = predicted_vector[z];
                predicted = z;
            }
        }

        int actual = test_labels[i];

        // Print just a few examples to keep console clean
        if (i < 10) { 
            std::cout << "Predicted: " << predicted << " | Actual: " << actual << "\n";
        }

        if (predicted == actual) correct++;
    }

    double accuracy = (double)correct / actual_test_size;
    std::cout << "\nAccuracy on " << actual_test_size << " test samples: " << accuracy * 100 << "%\n";
    
    return 0;
}