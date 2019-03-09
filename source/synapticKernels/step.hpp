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

#include <random>

#include "../synapticKernelHandler.hpp"

namespace hummus {
	class Neuron;
	
	class Step : public SynapticKernelHandler {
        
	public:
		// ----- CONSTRUCTOR -----
		Step(float _resetCurrent=5, float gaussianStandardDeviation=0) :
				resetCurrent(_resetCurrent),
				previousTimestamp(0) {
				
			// error handling
			if (resetCurrent <= 0) {
                throw std::logic_error("The current reset value cannot be less than or equal to 0");
            }
					
            // initialising a normal distribution
			std::random_device device;
            randomEngine = std::mt19937(device());
            normalDistribution = std::normal_distribution<>(0, gaussianStandardDeviation);
		}
		virtual ~Step(){}
		
		// ----- PUBLIC METHODS -----
		virtual double updateCurrent(double timestamp, float neuronCurrent) override {
			double updatedCurrent;
			if (timestamp - previousInputTime > resetCurrent) {
				updatedCurrent = 0;
			} else {
				updatedCurrent = neuronCurrent;
			}
			
			previousTimestamp = timestamp;
			
			return updatedCurrent;
		}
		
		virtual float integrateSpike(float neuronCurrent, float externalCurrent, double synapseWeight) override {
			return (neuronCurrent + externalCurrent * synapseWeight) + normalDistribution(randomEngine);
		}
		
	protected:
		float            		   resetCurrent;
		double            	 	   previousTimestamp;
		std::mt19937               randomEngine;
		std::normal_distribution<> normalDistribution;
	};
}
