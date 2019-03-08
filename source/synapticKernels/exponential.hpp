/*
 * dirac.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 23/01/2019
 *
 * Information: a synaptic kernel that instantly rises then exponentially decays
 */

#pragma once

#include "../synapticKernelHandler.hpp"

namespace hummus {
	class Neuron;
	
	class Dirac : public SynapticKernelHandler {
        
	public:
		// ----- CONSTRUCTOR -----
		Dirac() = default;
		virtual ~Dirac(){}
		
		// ----- PUBLIC METHODS -----
		virtual void updateCurrent(){}
	};
}
