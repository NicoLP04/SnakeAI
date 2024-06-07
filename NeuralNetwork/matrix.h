#pragma once
#include <iostream>
#include <vector>

class Matrix {
   public:
    Matrix() = default;
    Matrix(int height, int width);
    Matrix(std::vector<std::vector<double>> data);

    Matrix(const Matrix& other);
    Matrix& operator=(const Matrix& other);

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(double scalar) const;

    Matrix multiplyElementWise(const Matrix& other) const;
    Matrix dot(const Matrix& other) const;
    Matrix transpose() const;
    Matrix applyFunction(double (*function)(double)) const;

    int getHeight() const;
    int getWidth() const;
    double get(int i, int j) const;
    void set(int i, int j, double value);

    double sum() const;

    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);

   private:
    std::vector<std::vector<double>> mData;
    int mHeight;
    int mWidth;
};
