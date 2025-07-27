#pragma once

#include <vector>
#include <string>
#include <stdexcept>

class Matrix {
private:
    std::vector<std::vector<double>> data;
public:
    int rows, cols;

    Matrix();
    Matrix(int r, int c); // пустая матрица
    Matrix(const std::vector<std::vector<double>>& d); // из 2D-вектора

    // конструктор из вектора — создаёт матрицу-столбец (n x 1)
    Matrix(const std::vector<double>& vec);
    
    // доступ к элементу (i, j) с проверкой
    double& operator()(int i, int j) {
        if (i < 0 || i >= rows || j < 0 || j >= cols)
            throw std::out_of_range("индекс элемента вне допустимого диапазона");
        return data[i][j];
    }

    double operator()(int i, int j) const {
        if (i < 0 || i >= rows || j < 0 || j >= cols)
            throw std::out_of_range("индекс элемента вне допустимого диапазона");
        return data[i][j];
    }

    // доступ к строке с проверкой
    std::vector<double>& operator[](int i) {
        if (i < 0 || i >= rows)
            throw std::out_of_range("индекс строки вне допустимого диапазона");
        return data[i];
    }

    const std::vector<double>& operator[](int i) const {
        if (i < 0 || i >= rows)
            throw std::out_of_range("индекс строки вне допустимого диапазона");
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

    // создаёт матрицу из одинаковых значений
    static Matrix from_value(int rows, int cols, double val);

    // создаёт матрицу со случайными значениями
    static Matrix random(int rows, int cols, double min, double max);
    void resize(int new_rows, int new_cols);

    int getDataSize() const {
        return data.size();
    }

    void print(const std::string& name = "Matrix") const;
};