#include "Layer.h"
Layer::Layer(int size) {
	
	this->size = size;
	for (int i = 0; i < size; i++) {
		Neuron* n = new Neuron(0.00);
		this->neurons.push_back(n);
		
	}
}
void Layer::setVal(int i, double v) {
	this->neurons.at(i)->setVal(v);
}

Matrix *Layer::getValues() {

	Matrix* m = new Matrix(1, this->size, false);
	for (int i = 0; i < this->size; i++) {
		m->setValue(0, i, this->neurons.at(i)->getVal());
	}
	return m;
}

Matrix* Layer::getActivatedValues() {

	Matrix* m = new Matrix(1, this->size, false);
	for (int i = 0; i < this->size; i++) {
		m->setValue(0, i, this->neurons.at(i)->getActivatedVal());
	}
	return m;
}

Matrix* Layer::getDerivedValues() {

	Matrix* m = new Matrix(1, this->size, false);
	for (int i = 0; i < this->size; i++) {
		m->setValue(0, i, this->neurons.at(i)->getDerivedVal());
	}
	return m;
}

