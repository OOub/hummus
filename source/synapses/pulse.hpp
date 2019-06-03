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
		Pulse(Synapse* _target_neuron, Synapse* _parent_neuron, float _weight=1, float _delay=0, float _resetCurrent=5, float gaussianStandardDeviation=0) :
				Synapse(_target_neuron, _parent_neuron, _weight, _delay) {
			
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
		virtual ~Pulse(){}
		
		// ----- PUBLIC METHODS -----
		virtual double update(double timestamp, double timestep, float neuronCurrent) override {
            
			if (timestamp - previousInputTime > synapseTimeConstant) {
				return 0;
			} else {
				return neuronCurrent;
			}
		}
		
		virtual float receiveSpike(float neuronCurrent, float externalCurrent, float synapseWeight) override {
            return neuronCurrent + (externalCurrent+normalDistribution(randomEngine)) * synapseWeight;
		}
		
		virtual void toJson(nlohmann::json& output) override {
			// general synapse sparameters
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
