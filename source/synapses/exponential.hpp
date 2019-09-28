/*
 * dirac.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 23/01/2019
 *
 * Information: a synaptic kernel that instantly rises then exponentially decays
 * json_id 1
 */

#pragma once

#include <random>

#include "../synapse.hpp"
#include "../../third_party/json.hpp"

namespace hummus {
	class Neuron;
	
	class Exponential : public Synapse {
        
	public:
		// ----- CONSTRUCTOR -----
		Exponential(int _target_neuron, int _parent_neuron, float _weight, float _delay, float _synapse_time_constant=10, float _external_current=100, float _gaussian_std_dev=0) :
				Synapse(_target_neuron, _parent_neuron, _weight, _delay, _external_current) {
				
			synapse_time_constant = _synapse_time_constant;
            inv_s_tau = 1./synapse_time_constant;
                    
            gaussian_std_dev = _gaussian_std_dev;
			json_id = 1;
			
			// error handling
			if (_synapse_time_constant <= 0) {
                throw std::logic_error("The current decay value cannot be less than or equal to 0");
            }
				
			// initialising a normal distribution
            std::random_device device;
            random_engine = std::mt19937(device());
            normal_distribution = std::normal_distribution<float>(0, _gaussian_std_dev);
                    
            // current-based synapse figuring out if excitatory or inhibitory
            if (_weight < 0) {
                type = synapse_type::inhibitory;
            } else {
                type = synapse_type::excitatory;
            }
		}
		
		virtual ~Exponential(){}
		
		// ----- PUBLIC METHODS -----
        virtual float update(double timestamp, float timestep) override {
            // decay the current
            synaptic_current -= synaptic_current * timestep * inv_s_tau;
            return synaptic_current;
        }
        
		virtual void receive_spike() override {
            // increase the synaptic current in response to an incoming spike
            synaptic_current += weight * (external_current+normal_distribution(random_engine));
		}
        
		virtual void to_json(nlohmann::json& output) override {
			// general synapse parameters
            output.push_back({
                {"json_id", json_id},
                {"weight", weight},
                {"delay", delay},
                {"postsynaptic_neuron", postsynaptic_neuron},
                {"synapse_time_constant", synapse_time_constant},
            });
		}
		
	protected:
        float                           inv_s_tau;
		std::mt19937                    random_engine;
		std::normal_distribution<float> normal_distribution;
	};
}
