/*
 * dirac.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 23/01/2019
 *
 * Information: a synaptic kernel that instantly rises then exponentially decays
  * kernel type 1
 */

#pragma once

#include <random>

#include "../dependencies/json.hpp"
#include "../synapticKernelHandler.hpp"

namespace hummus {
	class Neuron;
	
	class Exponential : public SynapticKernelHandler {
        
	public:
		// ----- CONSTRUCTOR -----
		Exponential(float _decayCurrent=10, float gaussianStandardDeviation=0) :
				SynapticKernelHandler() {
				
			synapseTimeConstant = _decayCurrent;
			gaussianStdDev = gaussianStandardDeviation;
			type = 1;
			
			// error handling
			if (_decayCurrent <= 0) {
                throw std::logic_error("The current decay value cannot be less than or equal to 0");
            }
				
			// initialising a normal distribution
            std::random_device device;
            randomEngine = std::mt19937(device());
            normalDistribution = std::normal_distribution<>(0, gaussianStandardDeviation);
		}
		
		virtual ~Exponential(){}
		
		// ----- PUBLIC METHODS -----
		virtual double updateCurrent(double timestamp, double timestep, double previousInputTime, float neuronCurrent) override {
			
			// event-based
			if (timestep == 0) {
				return neuronCurrent * std::exp(-(timestep-previousInputTime)/synapseTimeConstant);
			// clock-based
			} else {
				return neuronCurrent * std::exp(-timestep/synapseTimeConstant);
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
				{"decayCurrent", synapseTimeConstant},
            });
		}
		
	protected:
		std::mt19937               randomEngine;
		std::normal_distribution<> normalDistribution;
	};
}
