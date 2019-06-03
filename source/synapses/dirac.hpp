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
		Dirac(Synapse* _target_neuron, Synapse* _parent_neuron, float _weight=1, float _delay=0, int _amplitudeScaling=50, float gaussianStandardDeviation=0) :
                Synapse(_target_neuron, _parent_neuron, _weight, _delay),
                amplitudeScaling(_amplitudeScaling) {
		
			gaussianStdDev = gaussianStandardDeviation;
			
			// initialising a normal distribution
			std::random_device device;
            randomEngine = std::mt19937(device());
            normalDistribution = std::normal_distribution<>(0, gaussianStandardDeviation);
		}
		
		virtual ~Dirac(){}
		
		// ----- PUBLIC METHODS -----
		virtual double update(double timestamp, double timestep, float neuronCurrent) override {
			return 0;
		}
		
		virtual float receiveSpike(float neuronCurrent, float externalCurrent, float synapseWeight) override {
            return amplitudeScaling * (neuronCurrent + (externalCurrent+normalDistribution(randomEngine)) * synapseWeight);
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
