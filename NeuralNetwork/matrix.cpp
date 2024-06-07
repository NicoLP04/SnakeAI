#include "matrix.h"

Matrix::Matrix(int height, int width) : mHeight(height), mWidth(width) {
    mData.resize(height, std::vector<double>(width, 0));
}

Matrix::Matrix(std::vector<std::vector<double>> data) : mData(data) {
    mHeight = data.size();
    mWidth = data[0].size();
}

Matrix::Matrix(const Matrix& other) {
    mHeight = other.getHeight();
    mWidth = other.getWidth();
    mData = other.mData;
}

Matrix& Matrix::operator=(const Matrix& other) {
    mHeight = other.getHeight();
    mWidth = other.getWidth();
    mData = other.mData;
    return *this;
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (mHeight != other.getHeight() || mWidth != other.getWidth()) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    Matrix result(mHeight, mWidth);
    for (int i = 0; i < mHeight; ++i) {
        for (int j = 0; j < mWidth; ++j) {
            result.set(i, j, mData[i][j] + other.get(i, j));
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (mHeight != other.getHeight() || mWidth != other.getWidth()) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    Matrix result(mHeight, mWidth);
    for (int i = 0; i < mHeight; ++i) {
        for (int j = 0; j < mWidth; ++j) {
            result.set(i, j, mData[i][j] - other.get(i, j));
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (mWidth != other.getHeight()) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    Matrix result(mHeight, other.getWidth());
    for (int i = 0; i < mHeight; ++i) {
        for (int j = 0; j < other.getWidth(); ++j) {
            double sum = 0;
            for (int k = 0; k < mWidth; ++k) {
                sum += mData[i][k] * other.get(k, j);
            }
            result.set(i, j, sum);
        }
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(mHeight, mWidth);
    for (int i = 0; i < mHeight; ++i) {
        for (int j = 0; j < mWidth; ++j) {
            result.set(i, j, mData[i][j] * scalar);
        }
    }
    return result;
}

Matrix Matrix::multiplyElementWise(const Matrix& other) const {
    if (mHeight != other.getHeight() || mWidth != other.getWidth()) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    Matrix result(mHeight, mWidth);
    for (int i = 0; i < mHeight; ++i) {
        for (int j = 0; j < mWidth; ++j) {
            result.set(i, j, mData[i][j] * other.get(i, j));
        }
    }
    return result;
}

Matrix Matrix::dot(const Matrix& other) const {
    if (mWidth != other.getHeight()) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    Matrix result(mHeight, other.getWidth());
    for (int i = 0; i < mHeight; ++i) {
        for (int j = 0; j < other.getWidth(); ++j) {
            double sum = 0;
            for (int k = 0; k < mWidth; ++k) {
                sum += mData[i][k] * other.get(k, j);
            }
            result.set(i, j, sum);
        }
    }
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(mWidth, mHeight);
    for (int i = 0; i < mHeight; ++i) {
        for (int j = 0; j < mWidth; ++j) {
            result.set(j, i, mData[i][j]);
        }
    }
    return result;
}

Matrix Matrix::applyFunction(double (*function)(double)) const {
    Matrix result(mHeight, mWidth);
    for (int i = 0; i < mHeight; ++i) {
        for (int j = 0; j < mWidth; ++j) {
            result.set(i, j, function(mData[i][j]));
        }
    }
    return result;
}

int Matrix::getHeight() const { return mHeight; }

int Matrix::getWidth() const { return mWidth; }

double Matrix::get(int i, int j) const { return mData[i][j]; }

void Matrix::set(int i, int j, double value) { mData[i][j] = value; }

double Matrix::sum() const {
    double sum = 0;
    for (int i = 0; i < mHeight; ++i) {
        for (int j = 0; j < mWidth; ++j) {
            sum += mData[i][j];
        }
    }
    return sum;
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    for (int i = 0; i < matrix.getHeight(); ++i) {
        for (int j = 0; j < matrix.getWidth(); ++j) {
            os << matrix.get(i, j) << " ";
        }
        os << std::endl;
    }
    return os;
}
