// MyNN.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "Neuron.h"
#include "Matrix.h"
#include "NeuralNetwork.h"
using namespace std;
int main()
{
    std::cout << "Hello World!\n";
    //Input
    Neuron *n1 = new Neuron(1.5);
    Neuron* n2 = new Neuron(0.5);
    Neuron* n3 = new Neuron(0.1);
    cout << "Val: " << n1->getVal() << endl;
    cout << "Activated val: " << n1->getActivatedVal() << endl;
    cout << "Derived val: " << n1->getDerivedVal() << endl;
    

    Matrix* m = new Matrix(3, 2, true);
    m->printToConsole();
    cout << "------------" << endl;
    Matrix* mT = m->transpose();
    mT->printToConsole();
    Matrix mProd = Matrix(*m) * Matrix(*mT);
    cout << "matrix multiplication" << endl;
    mProd.printToConsole();
    cout << "_______________" << endl;
    vector<int> topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(3);
    vector<double> input;
    input.push_back(1.0);
    input.push_back(0.0);
    input.push_back(1.0);

    NeuralNetwork* nn2 = new NeuralNetwork(topology);
    nn2->setCurrentInput(input);
    nn2->print();

    //Output
    Neuron* outputNeuron = new Neuron(0.0);


    vector<double> input2;
    input2.push_back(1);
    input2.push_back(0);
    input2.push_back(1);

    vector<int> topologyA;
    topologyA.push_back(3);
    topologyA.push_back(2);
    topologyA.push_back(1);
    cout << "_____________________" << endl;
    NeuralNetwork* nn = new NeuralNetwork(topologyA);
    nn2->setCurrentInput(input2);
    
    nn2->setCurrentTarget(input2);
    
    
    //training process
    for (int i = 0; i < 2000; i++) {

        cout << "Epoch: " << i << endl;
        nn2->feedForward();
        nn2->setErrors();
        //nn2->print();
        cout << "Total Error: " << nn2->getTotalError() << endl;
        nn2->backPropagate();
    }

    return 0;

}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
