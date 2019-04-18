/*
 * dirac.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 23/01/2019
 *
 * Information: instantly increase the current
 * kernel type 0
 */

#pragma once

#include <random>

#include "../dependencies/json.hpp"
#include "../synapticKernelHandler.hpp"

namespace hummus {
	class Neuron;
	
	class Dirac : public SynapticKernelHandler {
        
	public:
		// ----- CONSTRUCTOR -----
		Dirac(int _amplitudeScaling=50, float gaussianStandardDeviation=0) :
			amplitudeScaling(_amplitudeScaling),
			SynapticKernelHandler() {
		
			gaussianStdDev = gaussianStandardDeviation;
			
			// initialising a normal distribution
			std::random_device device;
            randomEngine = std::mt19937(device());
            normalDistribution = std::normal_distribution<>(0, gaussianStandardDeviation);
		}
		
		virtual ~Dirac(){}
		
		// ----- PUBLIC METHODS -----
		virtual double updateCurrent(double timestamp, double timestep, double previousInputTime, float neuronCurrent) override {
			return 0;
		}
		
		virtual float integrateSpike(float neuronCurrent, float externalCurrent, double synapseWeight) override {
            return amplitudeScaling * (neuronCurrent + (externalCurrent+normalDistribution(randomEngine)) * synapseWeight);
		}
	
		virtual void toJson(nlohmann::json& output) override {
			// general synaptic kernel parameters
            output.push_back({
            	{"type", type},
            	{"amplitudeScaling", amplitudeScaling},
				{"gaussianStdDev", gaussianStdDev},
            });
		}
		
	protected:
		int                        amplitudeScaling;
		std::mt19937               randomEngine;
		std::normal_distribution<> normalDistribution;
	};
}
