/*
 * step.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 23/01/2019
 *
 * Information: a synaptic kernel updating the current according to a step function; the current stays constant for a period of time then resets
 * kernel type 2
 */

#pragma once

#include <random>

#include "../dependencies/json.hpp"
#include "../synapticKernelHandler.hpp"

namespace hummus {
	class Neuron;
	
	class Step : public SynapticKernelHandler {
        
	public:
		// ----- CONSTRUCTOR -----
		Step(float _resetCurrent=5, float gaussianStandardDeviation=0) :
				SynapticKernelHandler() {
			
			synapseTimeConstant = _resetCurrent;
			gaussianStdDev = gaussianStandardDeviation;
			type = 2;
			
			// error handling
			if (_resetCurrent <= 0) {
                throw std::logic_error("The current reset value cannot be less than or equal to 0");
            }
					
            // initialising a normal distribution
			std::random_device device;
            randomEngine = std::mt19937(device());
            normalDistribution = std::normal_distribution<>(0, gaussianStandardDeviation);
		}
		virtual ~Step(){}
		
		// ----- PUBLIC METHODS -----
		virtual double updateCurrent(double timestamp, double timestep, double previousInputTime, float neuronCurrent) override {
            
			if (timestamp - previousInputTime > synapseTimeConstant) {
				return 0;
			} else {
				return neuronCurrent;
			}
		}
		
		virtual float integrateSpike(float neuronCurrent, float externalCurrent, double synapseWeight) override {
            return (neuronCurrent + (externalCurrent+normalDistribution(randomEngine)) * synapseWeight);
		}
		
		virtual void toJson(nlohmann::json& output) override {
			// general synaptic kernel parameters
            output.push_back({
            	{"type", type},
				{"gaussianStdDev", gaussianStdDev},
				{"resetCurrent", synapseTimeConstant},
            });
		}
		
	protected:
		std::mt19937               randomEngine;
		std::normal_distribution<> normalDistribution;
	};
}
