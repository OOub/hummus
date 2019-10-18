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
		Memristor(int _postsynaptic_neuron, int _presynaptic_neuron, float _weight) :
                Synapse(_postsynaptic_neuron, _presynaptic_neuron, _weight, 0, 0),
                V_syn(-1) {
            json_id = 3;
            type = synapse_type::excitatory;
		}

		virtual ~Memristor(){}

		// ----- PUBLIC METHODS -----
        virtual float update(double timestamp, float timestep) override {
            return 0;
        }

		virtual void receive_spike() override {
            // calculating synaptic current
            synaptic_current = weight * V_syn;
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
        float  V_syn;
	};
}
