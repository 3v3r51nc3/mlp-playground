#include "../include/matrix.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <random>

// m n * n p = m p
Matrix::Matrix() {

}

Matrix::Matrix(int r, int c) : rows(r), cols(c) {
    data.resize(rows, std::vector<double>(cols, 0.0));
}

Matrix::Matrix(const std::vector<std::vector<double>>& d) {
    data = d;
    rows = d.size();
    cols = d[0].size();
}

Matrix::Matrix(const std::vector<double>& vec) {
    rows = static_cast<int>(vec.size());
    cols = 1;
    data.resize(rows, std::vector<double>(cols, 0.0));
    for (int i = 0; i < rows; ++i) {
        data[i][0] = vec[i];
    }
}

void Matrix::print(const std::string& name) const {
    std::cout << name << " (" << rows << "x" << cols << "):\n";
    for (const auto& row : data) {
        for (double val : row) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << val << " ";
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(j, i) = data[i][j];
    return result;
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols)
        throw std::invalid_argument("matrix dimensions do not match for addition");

    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i, j) = data[i][j] + other(i, j);
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows)
        throw std::invalid_argument("incompatible dimensions for multiplication");

    Matrix result(rows, other.cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < other.cols; j++)
            for (int k = 0; k < cols; k++)
                result(i, j) += data[i][k] * other(k, j);
    return result;
}

Matrix Matrix::operator*(double x) const
{
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i, j) += data[i][j] * x;
    return result;
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this == &other)  // self-assignment protection
        return *this;

    rows = other.rows;
    cols = other.cols;
    data = other.data;  // std::vector handles data copying correctly

    return *this;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for operator+=");
    }
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            // assuming access via operator()
            data[r][c] += other(r, c);
        }
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions must match for operator+=");
    }
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            // assuming access via operator()
            data[r][c] -= other(r, c);
        }
    }
    return *this;
}

Matrix& Matrix::operator*=(double scalar) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            data[r][c] *= scalar;
        }
    }
    return *this;
}


Matrix Matrix::from_value(int rows, int cols, double val) {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result(i, j) = val;
    return result;
}

Matrix Matrix::random(int rows, int cols, double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min, max);

    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result(i, j) = dist(gen);
    return result;
}

void Matrix::resize(int new_rows, int new_cols) {
    // if dimensions don't change - do nothing
    if (new_rows == rows && new_cols == cols) return;

    // create a new container of the required size
    std::vector<std::vector<double>> new_data(new_rows, std::vector<double>(new_cols, 0.0));

    // copy old data to the new container (within the new bounds)
    int min_rows = std::min(rows, new_rows);
    int min_cols = std::min(cols, new_cols);
    for (int i = 0; i < min_rows; i++) {
        for (int j = 0; j < min_cols; j++) {
            new_data[i][j] = data[i][j];
        }
    }

    // replace old data with new data
    data = std::move(new_data);
    rows = new_rows;
    cols = new_cols;
}