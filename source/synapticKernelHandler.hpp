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
		SynapticKernelHandler() = default;
		
		virtual ~SynapticKernelHandler(){}
		
		// ----- PUBLIC METHODS -----
		
		// pure virtual method that updates the status of current before integrating a spike
        virtual double updateCurrent(double timestamp, float neuronCurrent) = 0;
		
        // pure virtual method that outputs an updated current value
		virtual float integrateSpike(float neuronCurrent, float externalCurrent, double synapseWeight) = 0;
	};
}

