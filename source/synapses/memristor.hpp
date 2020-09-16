/*
 * memristor.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 23/01/2019
 *
 * Information: a conductance-based synaptic kernel that reproduces the waveforms of the ULPEC memristor.
 */

#pragma once

#include <random>

#include "../synapse.hpp"

namespace hummus {
	class Neuron;

	class Memristor : public Synapse {

	public:
		// ----- CONSTRUCTOR -----
		Memristor(int _postsynaptic_neuron, int _presynaptic_neuron, double _weight, double _delay, double _current_sign=-1) :
                Synapse(_postsynaptic_neuron, _presynaptic_neuron, _weight, _delay),
                current_sign(_current_sign) {
            
            type = synapse_type::excitatory;
                    
            // initialising a normal distribution
            std::random_device device;
            random_engine = std::mt19937(device());
            normal_distribution = std::normal_distribution<float>(0, 0.1);
		}

		virtual ~Memristor(){}

		// ----- PUBLIC METHODS -----
		virtual void receive_spike(float potential=0) override {
            // updating synaptic_potential
            synaptic_potential += potential;
            // calculating synaptic current
            synaptic_current = current_sign * weight * synaptic_potential;
		}

        virtual void reset() override {
            synaptic_potential = 0;
            synaptic_current = 0;
        }
        
    protected:
        std::mt19937                     random_engine;
        std::normal_distribution<float>  normal_distribution;
        double                           current_sign;
	};
}
