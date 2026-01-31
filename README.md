# MLP Playground

A C++ implementation of a Multi-Layer Perceptron (MLP) built from first principles.

This project implements the core mathematical foundations of neural networks without relying on external machine learning frameworks such as PyTorch or TensorFlow. The implementation follows the concepts presented in *Grokking Deep Learning* by Andrew Trask and uses a custom matrix library for all linear algebra operations.

---

## Features

- **Small Math Library**  
  Standalone implementations of Matrix and Vector classes.

- **Configurable Architecture**  
  Supports variable layer sizes and multiple activation functions.

- **Manual Backpropagation**  
  Explicit implementation of the chain rule and weight updates.

- **MNIST Support**  
  Includes a loader for the MNIST handwritten digit dataset.

- **Performance-Oriented**  
  Uses OpenMP for multi-threaded CPU training.

---

## Build Instructions

The project uses **CMake** and requires a compiler supporting **C++17**.

### 1. Create a build directory

```bash
mkdir build
cd build
```

### 2. Configure the Project

(Release mode is recommended for performance.)

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
```

### 3. Compile

```bash
make
```

---

## Dataset

The MNIST dataset is required for training.

Download the following files from:  
http://yann.lecun.com/exdb/mnist/

Place them in a `datasets/` directory at the project root:

- train-images.idx3-ubyte  
- train-labels.idx1-ubyte  
- t10k-images.idx3-ubyte  
- t10k-labels.idx1-ubyte  

---

## Usage

After building the project and placing the dataset correctly, run:

```bash
./mlp-playground
```

The program will:

- Load the MNIST training and test datasets  
- Normalize input data  
- Train the neural network using Stochastic Gradient Descent  
- Output Mean Squared Error (MSE) during training  
- Evaluate and print final test accuracy  

---

## Notes

- The implementation is intended for educational purposes.
- All neural network logic is implemented manually without external ML libraries.
- Performance depends on compiler optimizations and OpenMP support.
