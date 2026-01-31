#pragma once

#include <vector>
#include <string>
#include <stdexcept>

class Matrix {
public:
    int rows, cols;

    Matrix();
    Matrix(int r, int c); // empty matrix
    Matrix(const std::vector<std::vector<double>>& d); // from 2D vector
    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), data(other.data) {} // copy constructor

    // constructor from vector - creates a column matrix (n x 1)
    Matrix(const std::vector<double>& vec);
    
    // access element (i, j) with bounds check
    double& operator()(int i, int j) {
        if (i < 0 || i >= rows || j < 0 || j >= cols)
            throw std::out_of_range("element index out of range");
        return data[i][j];
    }

    double operator()(int i, int j) const {
        if (i < 0 || i >= rows || j < 0 || j >= cols)
            throw std::out_of_range("element index out of range");
        return data[i][j];
    }

    // access row with bounds check
    std::vector<double>& operator[](int i) {
        if (i < 0 || i >= rows)
            throw std::out_of_range("row index out of range");
        return data[i];
    }

    const std::vector<double>& operator[](int i) const {
        if (i < 0 || i >= rows)
            throw std::out_of_range("row index out of range");
        return data[i];
    }

    Matrix transpose() const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(double x) const;
    Matrix& operator=(const Matrix& other);
    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);
    Matrix& operator*=(double scalar);

    // creates a matrix filled with a specific value
    static Matrix from_value(int rows, int cols, double val);

    // creates a matrix with random values
    static Matrix random(int rows, int cols, double min, double max);
    void resize(int new_rows, int new_cols);

    int getDataSize() const {
        return data.size();
    }

    void print(const std::string& name = "Matrix") const;
private:
    std::vector<std::vector<double>> data;
};