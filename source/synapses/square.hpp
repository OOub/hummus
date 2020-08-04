/*
 * square.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 23/01/2019
 *
 * Information: a current-based synaptic kernel updating the current according to a square pulse function. the current stays constant for a period of time then resets. only for neurons with current dynamics
 */

#pragma once

#include <random>

#include "../synapse.hpp"

namespace hummus {
	class Neuron;

	class Square : public Synapse {

	public:
		// ----- CONSTRUCTOR -----
		Square(int _target_neuron, int _parent_neuron, float _weight, float _delay, float _synapse_time_constant=10, float _external_current=80, float _gaussian_std_dev=0) :
				Synapse(_target_neuron, _parent_neuron, _weight, _delay),
                external_current(_external_current) {

            synapse_time_constant = _synapse_time_constant;

			// error handling
			if (_synapse_time_constant <= 0) {
                throw std::logic_error("The current reset value cannot be less than or equal to 0");
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
		virtual ~Square(){}

		// ----- PUBLIC METHODS -----
        virtual float update(double timestamp, float timestep=0) override {
            if (timestamp - previous_input_time > synapse_time_constant) {
                synaptic_current = 0;
            }
            return synaptic_current;
        }

		virtual void receive_spike(float potential=0) override {
            synaptic_current += efficacy * weight * (external_current+normal_distribution(random_engine));
		}

	protected:
		std::mt19937                     random_engine;
		std::normal_distribution<float>  normal_distribution;
        float                            external_current;
	};
}
