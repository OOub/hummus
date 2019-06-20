/*
 * dirac.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 02/06/2019
 *
 * Information: instantly increase the current
 * kernel type 0
 */

#pragma once

#include <random>

#include "../synapse.hpp"
#include "../dependencies/json.hpp"

namespace hummus {
	class Neuron;
	
	class Dirac : public Synapse {
        
	public:
		// ----- CONSTRUCTOR -----
		Dirac(int _target_neuron, int _parent_neuron, float _weight, float _delay, float _externalCurrent=100, int _amplitudeScaling=50, float gaussianStandardDeviation=0) :
                Synapse(_target_neuron, _parent_neuron, _weight, _delay, _externalCurrent),
                amplitudeScaling(_amplitudeScaling) {
		
			gaussianStdDev = gaussianStandardDeviation;
			
			// initialising a normal distribution
			std::random_device device;
            randomEngine = std::mt19937(device());
            normalDistribution = std::normal_distribution<>(0, gaussianStandardDeviation);
		}
		
		virtual ~Dirac(){}
		
		// ----- PUBLIC METHODS -----
        virtual float update(double timestamp) override {
            synapticCurrent = 0;
            return synapticCurrent;
        }
        
		virtual void receiveSpike(double timestamp) override {
            // saving timestamp
            previousInputTime = timestamp;
            synapticCurrent = amplitudeScaling * weight * (externalCurrent+normalDistribution(randomEngine));
		}
        
		virtual void toJson(nlohmann::json& output) override {
			// general synapse parameters
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
