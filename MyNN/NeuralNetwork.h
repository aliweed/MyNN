#pragma once
#include "Matrix.h"
#include "Layer.h"
#include <vector>
#include <iostream>
class NeuralNetwork
{
public:
	NeuralNetwork(vector<int> topology);
	void setCurrentInput(vector<double> input);
	void print();
	void feedForward();
	Matrix* getNeuronMatrix(int index);
	Matrix* getActivatedNeuronMatrix(int index);
	Matrix* getDerivedNeuronMatrix(int index);
	Matrix* getWeightMatrix(int index);
	void setNeuronValue(int indexLayer, int indexNeuron, double v)
	{
		this->layers.at(indexLayer)->setVal(indexNeuron, v);
	};

private:
	int					topologySize;
	vector<int>			topology;
	vector<Layer*>		layers;
	vector<Matrix*>		weightMatrices;
	vector<double>		input;
};
