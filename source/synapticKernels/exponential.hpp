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

#include <random>

#include "../synapticKernelHandler.hpp"

namespace hummus {
	class Neuron;
	
	class Exponential : public SynapticKernelHandler {
        
	public:
		// ----- CONSTRUCTOR -----
		Exponential(float _decayCurrent=10, float gaussianStandardDeviation=0) :
				decayCurrent(_decayCurrent),
				previousTimestamp(0) {
				
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
		virtual double updateCurrent(double timestamp, float neuronCurrent) override {
			double updatedCurrent = neuronCurrent * std::exp(-(timestamp - previousTimestamp)/decayCurrent);
			previousTimestamp = timestamp;
			
			return updatedCurrent;
		}
		
		virtual float integrateSpike(float neuronCurrent, float externalCurrent, double synapseWeight) override {
			return (neuronCurrent + externalCurrent * synapseWeight) + normalDistribution(randomEngine);
		}
	
	protected:
		float                      decayCurrent;
		double                     previousTimestamp;
		std::mt19937               randomEngine;
		std::normal_distribution<> normalDistribution;
	};
}
