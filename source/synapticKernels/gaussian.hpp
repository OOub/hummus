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
		Gaussian() = default;
		virtual ~Gaussian(){}
		
		// ----- PUBLIC METHODS -----
		virtual void updateCurrent(){}
	};
}
