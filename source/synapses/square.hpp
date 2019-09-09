/*
 * square.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 23/01/2019
 *
 * Information: a synaptic kernel updating the current according to a square pulse function; the current stays constant for a period of time then resets
 * json_id 2
 */

#pragma once

#include <random>

#include "../synapse.hpp"
#include "../../third_party/json.hpp"

namespace hummus {
	class Neuron;
	
	class Square : public Synapse {
        
	public:
		// ----- CONSTRUCTOR -----
		Square(int _target_neuron, int _parent_neuron, float _weight, float _delay, float _synapse_time_constant=5, float _external_current=150, float _gaussian_std_dev=0) :
				Synapse(_target_neuron, _parent_neuron, _weight, _delay, _external_current) {
			
            synapse_time_constant = _synapse_time_constant;
            gaussian_std_dev = _gaussian_std_dev;
			json_id = 2;
			
			// error handling
			if (_synapse_time_constant <= 0) {
                throw std::logic_error("The current reset value cannot be less than or equal to 0");
            }
					
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
		virtual ~Square(){}
		
		// ----- PUBLIC METHODS -----
        virtual float update(double timestamp) override {
            if (timestamp - previous_input_time > synapse_time_constant) {
                synaptic_current = 0;
            }
            return synaptic_current;
        }
        
		virtual void receive_spike(double timestamp) override {
            // saving timestamp
            previous_input_time = timestamp;
            synaptic_current += weight * (external_current+normal_distribution(random_engine));
		}
        
		virtual void to_json(nlohmann::json& output) override {
			// general synapse sparameters
            output.push_back({
                {"json_id", json_id},
                {"weight", weight},
                {"delay", delay},
                {"postsynaptic_neuron", postsynaptic_neuron},
				{"synapse_time_constant", synapse_time_constant},
            });
		}
		
	protected:
		std::mt19937               random_engine;
		std::normal_distribution<> normal_distribution;
	};
}
