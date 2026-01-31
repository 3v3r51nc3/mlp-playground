#pragma once
#include <vector>
#include "matrix.h"
#include "../include/vector_utils.h"

enum class GradientDescentType {
    Stochastic,    // SGD - update after every example
    MiniBatch,     // mini-batch - update after a group of examples
    Batch          // full batch - update after the entire epoch
};

struct LayerGradients {
    Matrix dW;
    std::vector<double> dB;
    std::vector<double> next_deltas; 
};

class Layer {
public:
    Layer(int input_count, int output_count, double dropout_rate, bool output = false, ActivationType activation = ActivationType::sigmoid, bool debug_info = false);

    std::vector<double> forward(const std::vector<double>& input);
    std::vector<double> backward(const std::vector<double>& input, const std::vector<double>& deltas, double learning_rate);

    void resize_weights(int input_count, int output_count);
    void print(const std::string& name) const;

    // --- new methods for gradient descent ---
    LayerGradients compute_gradients(const std::vector<double>& input, const std::vector<double>& deltas);
    void zero_grad_accum();
    void accumulate_gradients(const Matrix& dW, const std::vector<double>& dB);
    void apply_accumulated_gradients(double learning_rate);

    // getters
    int getInputSize() const { return weights.rows; }
    int getOutputSize() const { return weights.cols; }
    bool isOutputLayer() const { return is_output_layer; }
    void setDropoutEnabled(bool enabled) { dropout_enabled = enabled; }
private:
    Matrix weights;
    std::vector<double> biases;

    Matrix grad_weights_accum;
    std::vector<double> grad_biases_accum;
    int accum_count = 0;

    bool dropout_enabled = false;
    double dropout_rate;
    std::vector<bool> dropout_mask;
    void generate_dropout_mask(int size);

    std::vector<double> last_output;
    std::vector<double> last_pre_activation;

    ActivationType activation_type;
    bool is_output_layer;
    bool debug_info;
};