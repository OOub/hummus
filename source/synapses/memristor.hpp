/*
 * memristor.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 23/01/2019
 *
 * Information: a conductance-based synaptic kernel that reproduces the waveforms of the ULPEC memristor.
 * json_id 2
 */

#pragma once

#include <random>

#include "../synapse.hpp"
#include "../../third_party/json.hpp"

namespace hummus {
	class Neuron;

	class Memristor : public Synapse {

	public:
		// ----- CONSTRUCTOR -----
		Memristor(int _postsynaptic_neuron, int _presynaptic_neuron, double _weight, double _delay) :
                Synapse(_postsynaptic_neuron, _presynaptic_neuron, _weight, _delay) {
            json_id = 3;
            type = synapse_type::excitatory;
		}

		virtual ~Memristor(){}

		// ----- PUBLIC METHODS -----
		virtual void receive_spike(float potential=0) override {
            // updating synaptic_potential
            synaptic_potential += potential;
            
            // calculating synaptic current
            synaptic_current = weight * synaptic_potential;            
		}

        virtual void reset() override {
            synaptic_potential = 0;
            synaptic_current = 0;
        }
        
		virtual void to_json(nlohmann::json& output) override {
			// general synapse parameters
            output.push_back({
                {"json_id", json_id},
                {"weight", weight},
                {"delay", delay},
                {"postsynaptic_neuron", postsynaptic_neuron},
            });
		}
	};
}
