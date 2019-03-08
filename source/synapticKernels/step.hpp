/*
 * step.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 23/01/2019
 *
 * Information: a synaptic kernel updating the current according to a step function; the current stays constant for a period of time then resets
 */

#pragma once

#include "../synapticKernelHandler.hpp"

namespace hummus {
	class Neuron;
	
	class Step : public SynapticKernelHandler {
        
	public:
		// ----- CONSTRUCTOR -----
		Step() = default;
		virtual ~Step(){}
		
		// ----- PUBLIC METHODS -----
		virtual void updateCurrent(){}
	};
}
