/*
 * gaussian.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 23/01/2019
 *
 * Information: updates the current according to a gaussian distribution
 */

#pragma once

#include "../synapticKernelHandler.hpp"

namespace hummus {
	class Neuron;
	
	class Gaussian : public SynapticKernelHandler {
        
	public:
		// ----- CONSTRUCTOR -----
		Gaussian() : SynapticKernelHandler() {}
		virtual ~Gaussian(){}
		
		// ----- PUBLIC METHODS -----
		virtual float synapticIntegration(float neuronCurrent, float externalCurrent, double synapseWeight){}
	};
}
