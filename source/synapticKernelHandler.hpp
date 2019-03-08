/*
 * synapticKernelHandler.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 25/09/2018
 *
 * Information: The SynapticKernelHandler class is called from a neuron to apply a synaptic kernel and update the current
 */

#pragma once

namespace hummus {
	class Network;
	class Neuron;
	
	class SynapticKernelHandler {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		SynapticKernelHandler() : previousInputTime(0) {}
		
		virtual ~SynapticKernelHandler(){}
		
		// ----- PUBLIC METHODS -----
        
        // pure virtual function that needs to be implemented in every synaptic kernel, and output an updated current value
		virtual float synapticIntegration(float neuronCurrent, float externalCurrent, double synapseWeight) = 0;
	
	protected:
	
		double previousInputTime;
	};
}

