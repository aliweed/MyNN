#pragma once
#include "Matrix.h"
#include "Layer.h"
#include <vector>
#include <iostream>
#include <assert.h>
#include <algorithm>
class NeuralNetwork
{
public:
	NeuralNetwork(vector<int> topology);
	void setCurrentInput(vector<double> input);
	void setCurrentTarget(vector<double> target);
	void print();
	void feedForward();
	void backPropagate();
	Matrix* getNeuronMatrix(int index);
	Matrix* getActivatedNeuronMatrix(int index);
	Matrix* getDerivedNeuronMatrix(int index);
	Matrix* getWeightMatrix(int index);

	void setErrors();
	void setNeuronValue(int indexLayer, int indexNeuron, double v){
		this->layers.at(indexLayer)->setVal(indexNeuron, v);
	}
	double getTotalError() { return this->error; };
	vector<double> getErrors() { return this->errors; };

private:
	int					topologySize;
	vector<int>			topology;
	vector<Layer*>		layers;
	vector<Matrix*>		weightMatrices;
	vector<double>		input;
	vector<double>		target;
	double				error;
	vector<double>		errors;
	vector<double>		historicalErrors;
};
