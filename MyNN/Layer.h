#pragma once
#include <iostream>
#include "Neuron.h"
#include <vector>
#include "Matrix.h"
using namespace std;
class Layer
{
public:
	Layer(int size);
	void setVal(int i, double v);
	Matrix *getValues();
	Matrix *getActivatedValues();
	Matrix *getDerivedValues();

	
private:
	int size;

	vector<Neuron*> neurons;

};

