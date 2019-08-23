/*
 * dirac.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 02/06/2019
 *
 * Information: instantly increase the current
 * json_id 0
 */

#pragma once

#include <random>

#include "../synapse.hpp"
#include "../../third_party/json.hpp"

namespace hummus {
	class Neuron;
	
	class Dirac : public Synapse {
        
	public:
		// ----- CONSTRUCTOR -----
		Dirac(int _target_neuron, int _parent_neuron, float _weight, float _delay, float _amplitudeScaling=50, float _externalCurrent=100, float gaussianStandardDeviation=0) :
                Synapse(_target_neuron, _parent_neuron, _weight, _delay, _externalCurrent),
                amplitudeScaling(_amplitudeScaling) {
		
			gaussianStdDev = gaussianStandardDeviation;
			
			// initialising a normal distribution
			std::random_device device;
            randomEngine = std::mt19937(device());
            normalDistribution = std::normal_distribution<>(0, gaussianStandardDeviation);
            
            // current-based synapse figuring out if excitatory or inhibitory
            if (_weight < 0) {
                type = synapseType::inhibitory;
            } else {
                type = synapseType::excitatory;
            }
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
            	{"json_id", json_id},
                {"weight", weight},
                {"delay", delay},
                {"postsynapticNeuron", postsynaptic_neuron},
            	{"amplitudeScaling", amplitudeScaling},
            });
		}
		
	protected:
		double                     amplitudeScaling;
		std::mt19937               randomEngine;
		std::normal_distribution<> normalDistribution;
        int                        json_synapse_type;
	};
}
