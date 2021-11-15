#include "NeuralNetwork.h"
NeuralNetwork::NeuralNetwork(vector<int> topology) {
	this->topology =		topology;
	this->topologySize =	topology.size();

	for (int i  = 0; i < topologySize; i++) {
		Layer* l = new Layer(topology.at(i));
		this->layers.push_back(l);
	}
	for (int i = 0; i < topologySize - 1; i++) {
		Matrix* m = new Matrix(topology.at(i), topology.at(i + 1), true); // rows = no. of neurons in this layer, cols = no. of rows in next layer.
		this->weightMatrices.push_back(m);
	}
}

void NeuralNetwork::setCurrentInput(vector<double> input) {
	this->input = input;
	for (int i = 0; i < input.size(); i++) {
		this->layers.at(0)->setVal(i, input.at(i));
	}
}
void NeuralNetwork::setCurrentTarget(vector<double> target) {
	this->target = target;
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
void NeuralNetwork::setErrors() {
	double targetSize = target.size();
	double layerCount = this->layers.size();
	Layer outputLayer = *this->layers.at(layerCount - 1);
	vector<Neuron*> outputNeurons = outputLayer.getNeurons();
	double outputLayerSize = outputNeurons.size();
	errors.clear();

	if (targetSize == 0) {
		cerr << "Target size " << targetSize << " is not the same as output layer size:" << 
			outputLayerSize << endl;
		assert(false);
	}
	if (targetSize != outputLayerSize) {
		cerr << "Target size " << targetSize << " is not the same as output layer size: " << outputLayerSize << endl;
		assert(false);
	}


	this->error = 0.00;

	
	for (int i = 0; i < targetSize; i++) {
		double tempErr = outputNeurons.at(i)->getActivatedVal() - target.at(i); // simple cost function
		errors.push_back(tempErr);
		this->error += tempErr;
	}
	historicalErrors.push_back(this->error);
}
void NeuralNetwork::backPropagate() {

	vector<Matrix*> newWeights;
	Matrix* gradient;
	int targetSize = target.size();
	int layerCount = this->layers.size();
	Layer outputLayer = *this->layers.at(layerCount - 1);
	vector<Neuron*> outputNeurons = outputLayer.getNeurons();
	double outputLayerSize = outputNeurons.size();

	//**********compute gradients for the output layer ********
	// ********************************************************
	// ********************************************************
	//*****gradient = [output layer derived value] * [error]***
	// ********************************************************
	// ********************************************************
	// ********************************************************
	Matrix* derivedValuesOutput = outputLayer.getDerivedValues(); // derived values of output layer
	Matrix* gradientsOutput = new Matrix(1, outputNeurons.size(), false); // setup gradient matrix
	for (int i = 0; i < this->errors.size(); i++) {
		double d = derivedValuesOutput->getValue(0, i);
		double e = this->errors.at(i);
		double g = d * e; // gradient = [output layer derived value] * [error]
		gradientsOutput->setValue(0, i, g); // compute gradients
	}

	//**********compute new weights for last hidden layer*****************************
	// *****************************************************************************
	// *****************************************************************************
	//*****delta = [gradients from output layer] * [last hidden activated values]***
	// [new weights] = weights - delta
	// *****************************************************************************
	// *****************************************************************************
	// *****************************************************************************
	int lastHiddenLayerIndex = layerCount - 2;
	Layer* lastHiddenLayer = this->layers.at(lastHiddenLayerIndex);
	Matrix* lastHiddenActivated = this->layers.at(lastHiddenLayerIndex)->getActivatedValues();
	//delta = [gradients from output layer] * [last hidden activated values]
	Matrix* lastHiddenDelta = (Matrix(*gradientsOutput->transpose()) * Matrix(*lastHiddenActivated)).transpose();


	Matrix* lastHiddenWeights = this->weightMatrices.at(lastHiddenLayerIndex);
	Matrix* lastHiddenWeightsNew = new Matrix(lastHiddenDelta->getNumRows(),
										lastHiddenDelta->getNumCols(),
										false);

	for (int row = 0; row < lastHiddenDelta->getNumRows(); row++) {
		for (int col = 0; col < lastHiddenDelta->getNumCols(); col++) {
			double originalWeight = lastHiddenWeights->getValue(row, col);
			double deltaWeight = lastHiddenDelta->getValue(row, col);
			lastHiddenWeightsNew->setValue(row, col, originalWeight - deltaWeight);
		}
	}
	newWeights.push_back(lastHiddenWeightsNew);
	gradient = new Matrix(gradientsOutput->getNumRows(), gradientsOutput->getNumCols(), false);
	for (int row = 0; row < gradientsOutput->getNumRows(); row++) {
		for (int col = 0; col < gradientsOutput->getNumCols(); col++) {
			gradient->setValue(row, col, gradientsOutput->getValue(row, col));
		}
	}

	cout << "Output to hidden new weights" << endl;
	lastHiddenWeightsNew->printToConsole();
	//**********	Last hidden layer to input layer   *****************************
	// *****************************************************************************
	// *****************************************************************************
	for (int i = lastHiddenLayerIndex; i > 0; i--) {
		Layer* thisLayer = this->layers.at(i);
		Matrix* thisLayerDerived = thisLayer->getDerivedValues();
		Matrix* thisLayerDerivedGradient = new Matrix(1, thisLayer->getNeurons().size(), false);
		Matrix* thisLayerActivatedVals = thisLayer->getActivatedValues();
		Matrix* weightMatrix = this->weightMatrices.at(i);
		Matrix* leftNeurons = this->layers.at(0)->getValues();
		Matrix* weightMatrixL = this->weightMatrices.at(i - 1);
		if (i > 1) {
			leftNeurons = this->layers.at(i - 1)->getActivatedValues();
		}

		for (int row = 0; row < weightMatrix->getNumRows(); row++) {
			double sum = 0;
			for (int col = 0; col < weightMatrix->getNumCols(); col++) {
				double p = gradient->getValue(0, col) * weightMatrix->getValue(row, col);
				sum += p;
			}
			double g = sum * thisLayerActivatedVals->getValue(0, row);
			thisLayerDerivedGradient->setValue(0, row, g);
		}
			
		Matrix* deltaWeights = (*thisLayerDerivedGradient->transpose() * Matrix (*leftNeurons)).transpose();
		Matrix* newWeightsHidden = new Matrix(deltaWeights->getNumRows(), deltaWeights->getNumCols(), false);
		for (int row = 0; row < newWeightsHidden->getNumRows(); row++) {

			for (int col = 0; col < newWeightsHidden->getNumCols(); col++) {
				double w = weightMatrixL->getValue(row, col);
				double d = deltaWeights->getValue(row, col);
				double n = w - d;
				newWeightsHidden->setValue(row, col, n);
			}
		}
		newWeights.push_back(newWeightsHidden);
		gradient = new Matrix(thisLayerDerivedGradient->getNumRows(), thisLayerDerivedGradient->getNumCols(), false);
		for (int row = 0; row < thisLayerDerivedGradient->getNumRows(); row++) {
			for (int col = 0; col < thisLayerDerivedGradient->getNumCols(); col++) {
				gradient->setValue(row, col, thisLayerDerivedGradient->getValue(row, col));
			}
		}
		 

	}
	
	std::reverse(newWeights.begin(), newWeights.end());
	this->weightMatrices = newWeights;
	


}