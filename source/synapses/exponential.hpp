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

#include "../synapse.hpp"
#include "../dependencies/json.hpp"

namespace hummus {
	class Neuron;
	
	class Exponential : public Synapse {
        
	public:
		// ----- CONSTRUCTOR -----
		Exponential(int _target_neuron, int _parent_neuron, float _weight, float _delay, float _synapseTimeConstant=50, float _externalCurrent=250, float gaussianStandardDeviation=0) :
				Synapse(_target_neuron, _parent_neuron, _weight, _delay, _externalCurrent) {
				
			synapseTimeConstant = _synapseTimeConstant;
			gaussianStdDev = gaussianStandardDeviation;
			type = 1;
			
			// error handling
			if (_synapseTimeConstant <= 0) {
                throw std::logic_error("The current decay value cannot be less than or equal to 0");
            }
				
			// initialising a normal distribution
            std::random_device device;
            randomEngine = std::mt19937(device());
            normalDistribution = std::normal_distribution<>(0, gaussianStandardDeviation);
		}
		
		virtual ~Exponential(){}
		
		// ----- PUBLIC METHODS -----
        virtual float update(double timestamp) override {
            // exponentially decay the current
            synapticCurrent = synapticCurrent * std::exp(-(timestamp - previousInputTime)/synapseTimeConstant);
            return synapticCurrent;
        }
        
		virtual void receiveSpike(double timestamp) override {
            // saving timestamp
            previousInputTime = timestamp;
            
            // increase the synaptic current in response to an incoming spike
            synapticCurrent += weight * (externalCurrent+normalDistribution(randomEngine));
		}
        
		virtual void toJson(nlohmann::json& output) override {
			// general synapse parameters
            output.push_back({
                {"type", type},
                {"weight", weight},
                {"delay", delay},
                {"postsynapticNeuron", postsynaptic_neuron},
				{"synapseTimeConstant", synapseTimeConstant},
            });
		}
		
	protected:
		std::mt19937               randomEngine;
		std::normal_distribution<> normalDistribution;
	};
}
