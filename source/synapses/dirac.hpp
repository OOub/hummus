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
		Dirac(int _target_neuron, int _parent_neuron, float _weight, float _delay, float _amplitude_scaling=50, float _external_current=150, float _gaussian_std_dev=0) :
                Synapse(_target_neuron, _parent_neuron, _weight, _delay, _external_current),
                amplitude_scaling(_amplitude_scaling) {
		
            gaussian_std_dev = _gaussian_std_dev;
			
			// initialising a normal distribution
			std::random_device device;
            random_engine = std::mt19937(device());
            normal_distribution = std::normal_distribution<>(0, _gaussian_std_dev);
            
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
            synaptic_current = 0;
            return synaptic_current;
        }
        
		virtual void receive_spike(double timestamp) override {
            // saving timestamp
            previous_input_time = timestamp;
            synaptic_current = amplitude_scaling * weight * (external_current+normal_distribution(random_engine));
		}
        
		virtual void to_json(nlohmann::json& output) override {
			// general synapse parameters
            output.push_back({
            	{"json_id", json_id},
                {"weight", weight},
                {"delay", delay},
                {"postsynaptic_neuron", postsynaptic_neuron},
            	{"amplitude_scaling", amplitude_scaling},
            });
		}
		
	protected:
		double                     amplitude_scaling;
		std::mt19937               random_engine;
		std::normal_distribution<> normal_distribution;
        int                        json_synapse_type;
	};
}
