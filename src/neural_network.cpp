#include <iostream>
#include <iomanip>
#include <cassert>

#include "../include/neural_network.h"
#include "../include/vector_utils.h"

NeuralNetwork::NeuralNetwork(Matrix input, Matrix target, double learning_rate, double dropout_rate, double mse_stop_point) : inputs(input), 
targets(target), dropout_rate(dropout_rate), learning_rate(learning_rate), mse_stop_point(mse_stop_point) {

}

NeuralNetwork::~NeuralNetwork() {

}

std::string activationToString(ActivationType act) {
    switch (act) {
    case ActivationType::sigmoid: return "sigmoid";
    case ActivationType::relu: return "relu";
    case ActivationType::leaky_relu: return "leaky relu";
    case ActivationType::linear: return "linear";
    default: return "unknown";
    }
}

std::string gdTypeToString(GradientDescentType gdt) {
    switch (gdt) {
    case GradientDescentType::Stochastic: return "stochastic";
    case GradientDescentType::Batch: return "batch";
    case GradientDescentType::MiniBatch: return "mini-batch";
    default: return "unknown";
    }
}


void NeuralNetwork::train(int epoch_times, GradientDescentType gd_type, int mini_batch_size) {
    std::cout << "Training started!\n\n";

    std::cout << "Network structure:\n";

    for (size_t i = 0; i < layers.size(); ++i) {
        const Layer& layer = layers[i];
        int input_size = layer.getInputSize();
        int output_size = layer.getOutputSize();

        if (i == 0) {
            std::cout << "(input) " << input_size << " -> " << output_size;
        }
        else {
            std::cout << " -> " << output_size;
        }

        if (layer.isOutputLayer()) {
            std::cout << " (output)";
        }
    }
    std::cout << "\n\n";


    std::cout << "Network params:\n";

    std::cout << "  sample count: " << inputs.rows << "\n";
    std::cout << "  learning rate: " << learning_rate << "\n";
    std::cout << "  dropout rate: " << dropout_rate << "\n";
    std::cout << "  gradient descent type: " << gdTypeToString(gd_type) << "\n";
    if (gd_type == GradientDescentType::MiniBatch) std::cout << "   mini-batch size: " << mini_batch_size << "\n";
    std::cout << "  epoch count: " << epoch_times << "\n";
    std::cout << "  mse_stop_point: " << mse_stop_point << "\n\n";


    int print_every_n_epochs = 1;
    if (epoch_times > 10000) print_every_n_epochs = 1000;
    else if (epoch_times > 1000) print_every_n_epochs = 100;
    else if (epoch_times > 100) print_every_n_epochs = 10;

    for (int epoch = 0; epoch < epoch_times; epoch++) {
        double epoch_error_sum = 0.0;

        if (gd_type != GradientDescentType::Stochastic) {
            // for batch methods, reset gradient accumulation on all layers
            for (auto& layer : layers) {
                layer.zero_grad_accum();
            }
        }

        int sample_count = inputs.rows;

        int step = 1;
        if (gd_type == GradientDescentType::MiniBatch) step = mini_batch_size;
        else if (gd_type == GradientDescentType::Batch) step = sample_count;

        for (int start = 0; start < sample_count; start += step) {
            int batch_end = std::min(start + step, sample_count);

            // for each example in the batch
            for (int i = start; i < batch_end; ++i) {
                auto input = inputs[i];
                auto target = targets[i];

                // forward pass
                std::vector<std::vector<double>> activations;
                activations.push_back(input); //first input layer is never activated
                for (auto& layer : layers) {
                    layer.setDropoutEnabled(!layer.isOutputLayer() && dropout_rate > 0);
                    //layer.setDropoutEnabled(false);

                    input = layer.forward(input);
                    activations.push_back(input);
                }
                auto prediction = input;

                // calculate deltas for output
                std::vector<double> deltas(prediction.size());
                double sample_mse = 0.0;

                for (size_t d = 0; d < deltas.size(); ++d) {
                    deltas[d] = targets[i][d] - prediction[d];
                    sample_mse += deltas[d] * deltas[d];
                }

                sample_mse /= deltas.size();
                epoch_error_sum += sample_mse;

                // backward pass
                for (int l = layers.size() - 1; l >= 0; --l) {
                    auto& layer = layers[l];
                    auto input_to_layer = activations[l];

                    if (gd_type == GradientDescentType::Stochastic) {
                        // update immediately
                        deltas = layer.backward(input_to_layer, deltas, learning_rate);
                    }
                    else {
                        // return gradients but do not update weights
                        auto grads = layer.compute_gradients(input_to_layer, deltas); // 0.0 to avoid updating
                        // need to accumulate gradients in layer (you need to modify backward or add accumulation methods)
                        layer.accumulate_gradients(grads.dW, grads.dB); // pass dW, dB here
                        deltas = grads.next_deltas;
                    }
                }
            }
            // after batch processing
            if (gd_type != GradientDescentType::Stochastic) {
                // apply accumulated gradients after batch
                for (auto& layer : layers)
                    layer.apply_accumulated_gradients(learning_rate);
            }
        }

        double epoch_mse = epoch_error_sum / sample_count;
        if ((epoch + 1) % print_every_n_epochs == 0 || epoch == 0 || epoch_mse < 0.0001) {
            std::cout << "epoch " << epoch + 1 << " | mse = " << epoch_mse << "\n";
        }
        if (epoch_mse < mse_stop_point) {
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "MSE is too low (" << epoch_mse << "), no need to continue training... Quitting cycle" << "\n";
            std::cout.unsetf(std::ios::fixed); // to return to normal output mode
            std::cout << std::setprecision(2); // can reset if needed
            break;
        }
    }

    std::cout << "\ntraining complete!\n";
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    assert(input.size() == static_cast<size_t>(inputs.cols));
    std::vector<double> output = input;

    if (debug_info) VectorUtils::print(output, "input");
    for (auto& layer : layers) {
        layer.setDropoutEnabled(false);
        output = layer.forward(output);
    }

    if (debug_info) VectorUtils::print(output, "prediction");

    return output;
}

Layer& NeuralNetwork::add_layer(int input_size, int output_size, bool is_output, ActivationType activation) {
    Layer new_layer(input_size, output_size, dropout_rate, is_output, activation);
    layers.push_back(new_layer);

    return layers[layers.size() - 1];
}
