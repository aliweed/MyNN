#include "Matrix.h"
using namespace std;
Matrix::Matrix(int numRows, int numCols, bool isRandom) {
	this->numRows = numRows;
	this->numCols = numCols;
	for (int i = 0; i < numRows; i++) {
		vector<double> colValues;
		for (int j = 0; j < numCols; j++) {
			double r = 0.00;
			if (isRandom) {
				r = this->generateRandomNumber();

			}
			colValues.push_back(r);

		}
		this->values.push_back(colValues);
	}
}
double Matrix::generateRandomNumber() {
	random_device rd;
	mt19937 gen(rd());
	uniform_real_distribution<> dis(0, 1);
	return dis(gen);
}
void Matrix::printToConsole() {
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			cout << this->values.at(i).at(j) << "\t\t";
		}
		cout << endl;
	}
}

Matrix *Matrix::transpose() {
	Matrix* m = new Matrix(this->numCols, this->numRows, false);
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			m->setValue(j, i, this->getValue(i, j));
		}
	}
	return m;
}

Matrix Matrix::operator* ( const Matrix& m1) {
	
	if (this->getNumCols() != m1.getNumRows()) {
		throw new exception("matrix dimensions mismatch");
	}

	Matrix m = Matrix(this->getNumRows(), m1.getNumCols(), false);
	



	for (int row = 0; row < this->getNumRows(); row++) {
		for (int col = 0; col < m1.getNumCols(); col++) {
			// Multiply the row of A by the column of B to get the row, column of product.
			for (int inner = 0; inner < m1.getNumRows(); inner++) {
				double p = this->getValue(row, inner) * m1.getValue(inner, col);
				double newVal = m.getValue(row, col) + p;
				m.setValue(row, col, newVal);
			}			
		}
	}
	return m;
}