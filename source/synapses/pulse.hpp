/*
 * pulse.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 23/01/2019
 *
 * Information: a synaptic kernel updating the current according to a square pulse function; the current stays constant for a period of time then resets
 * kernel type 2
 */

#pragma once

#include <random>

#include "../synapse.hpp"
#include "../dependencies/json.hpp"

namespace hummus {
	class Neuron;
	
	class Pulse : public Synapse {
        
	public:
		// ----- CONSTRUCTOR -----
		Pulse(int _target_neuron, int _parent_neuron, float _weight, float _delay, float _synapseTimeConstant=5, float _externalCurrent=100, float gaussianStandardDeviation=0) :
				Synapse(_target_neuron, _parent_neuron, _weight, _delay, _externalCurrent) {
			
			synapseTimeConstant = _synapseTimeConstant;
			gaussianStdDev = gaussianStandardDeviation;
			type = 2;
			
			// error handling
			if (_synapseTimeConstant <= 0) {
                throw std::logic_error("The current reset value cannot be less than or equal to 0");
            }
					
            // initialising a normal distribution
			std::random_device device;
            randomEngine = std::mt19937(device());
            normalDistribution = std::normal_distribution<>(0, gaussianStandardDeviation);
		}
		virtual ~Pulse(){}
		
		// ----- PUBLIC METHODS -----
        virtual float update(double timestamp) override {
            if (timestamp - previousInputTime > synapseTimeConstant) {
                synapticCurrent = 0;
            }
            return synapticCurrent;
        }
        
		virtual void receiveSpike(double timestamp) override {
            // saving timestamp
            previousInputTime = timestamp;
            synapticCurrent += weight * (externalCurrent+normalDistribution(randomEngine));
		}
        
		virtual void toJson(nlohmann::json& output) override {
			// general synapse sparameters
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
