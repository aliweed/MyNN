#pragma once
#include <vector>
#include <random>
#include <iostream>
using namespace std;
class Matrix
{
	

public:
	Matrix operator *  (const Matrix& m1);
	Matrix(int numRows, int numCols, bool isRandom);
	Matrix *transpose();
	void setValue(int r, int c, double v) {  this->values.at(r).at(c) = v; }
	double getValue(int r, int c) const { return this->values.at(r).at(c); }
	double generateRandomNumber();
	int getNumRows() const { return this->numRows; }
	int getNumCols() const { return this->numCols; }
	void printToConsole();
	void multiply(Matrix *m);

private:
	vector< vector<double> > values;
	int numRows;
	int numCols;
};

