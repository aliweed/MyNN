#include "NeuralNetwork.h"
NeuralNetwork::NeuralNetwork(vector<int> topology) {
	this->topology =		topology;
	this->topologySize =	topology.size();

	for (int i  = 0; i < topologySize; i++) {
		Layer* l = new Layer(topology.at(i));
		this->layers.push_back(l);
	}
	for (int i = 0; i < topologySize - 1; i++) {
		Matrix* m = new Matrix(topology.at(i), topology.at(i + 1), true);
		this->weightMatrices.push_back(m);
	}
}

void NeuralNetwork::setCurrentInput(vector<double> input) {
	this->input = input;
	for (int i = 0; i < input.size(); i++) {
		this->layers.at(0)->setVal(i, input.at(i));
	}
}
void NeuralNetwork::print() {
	cout << "================             NETWORK PRINT            ================" << endl;
	for(int i = 0; i < this->layers.size(); i++){
		cout << "LAYER(" << i+1 <<") Values:" << endl;
		if (i == 0) {
			Matrix* m = this->layers[0]->getValues();
			m->printToConsole();
		}
		else {
			Matrix* m = this->layers[i]->getActivatedValues();
			m->printToConsole();
		}
		if (i < this->layers.size() - 1) {
			cout << "LAYER(" << i + 1 << ") Weights:" << endl;
			this->getWeightMatrix(i)->printToConsole();
			cout << "______________________________________________" << endl;
		}
	}
	cout << "==================================================" << endl;
}  
void NeuralNetwork::feedForward() {
	for (int i = 0; i < (this->layers.size() - 1); i++) {
		Matrix* a = this->getNeuronMatrix(i);
		if (i != 0) {
			a = this->getActivatedNeuronMatrix(i);
		}
		Matrix* b = this->getWeightMatrix(i);
		Matrix c = Matrix(*a) * Matrix(*b);
		for (int c_index = 0; c_index < c.getNumCols(); c_index++) {
			this->setNeuronValue(i + 1, c_index, c.getValue(0, c_index));
		} 
	}
}
Matrix* NeuralNetwork::getNeuronMatrix(int index)
{
	return this->layers.at(index)->getValues();
}
Matrix* NeuralNetwork::getActivatedNeuronMatrix(int index) {
	return this->layers.at(index)->getActivatedValues();
}
Matrix* NeuralNetwork::getDerivedNeuronMatrix(int index) {
	return this->layers.at(index)->getDerivedValues();
}
Matrix* NeuralNetwork::getWeightMatrix(int index) {
	return this->weightMatrices.at(index);
}
