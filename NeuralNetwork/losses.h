#pragma once

#include <cmath>

#include "matrix.h"

double mse(const Matrix& predictions, const Matrix& targets);
Matrix msePrime(const Matrix& predictions, const Matrix& targets);

double crossEntropy(const Matrix& predictions, const Matrix& targets);
Matrix crossEntropyPrime(const Matrix& predictions, const Matrix& targets);
