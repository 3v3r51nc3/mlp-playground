#pragma once

#include <vector>
#include <string>

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

    double& at(int i, int j);
    double at(int i, int j) const;

    std::vector<double> getRow(int i) const;
    void setRow(int i, const std::vector<double>& row);

    void setValue(int i, int j, double value);
    void addValue(int i, int j, double value);


    Matrix transpose() const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix& operator=(const Matrix& other);

    double& operator()(int i, int j) {
        return data[i][j];
    }

    double operator()(int i, int j) const {
        return data[i][j];
    }

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