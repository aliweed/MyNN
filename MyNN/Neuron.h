#ifndef _NEURON_H_
#define _NEURON_H_
#pragma once
#include <iostream>
#include <math.h>
using namespace std;
class Neuron
{
public:	
	Neuron(double val);
	void setVal(double val);

	//Fast Sigmoid Function
	//f(x) = x / (1 + |x|) 
	//|x| - absolute value
	void activate();

	//derivative for fast sigmoid function 
	//f'(x) = f(x) * (1 - f(x))
	void derive();

	double getVal() { return this->val; }
	double getActivatedVal() { return this->activatedVal; }
	double getDerivedVal() { return this->derivedVal; }
private:
	double val;
	double activatedVal; //~normalized val
	double derivedVal; //approximated derivative of the activated value
};

#endif
