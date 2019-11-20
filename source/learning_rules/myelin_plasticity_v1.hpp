/*
 * myelin_plasticity_v1.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: The MP_1 learning rule
 */

#pragma once

#include <stdexcept>
#define _USE_MATH_DEFINES

#include "../addon.hpp"
#include "../addons/myelin_plasticity_logger.hpp"

namespace hummus {
    class Synapse;
	class Neuron;
	
	class MP_1 : public Addon {
        
	public:
		// ----- CONSTRUCTOR -----
        MP_1(int _time_constant=10, float _learning_rate=1) :
                time_constant(_time_constant),
                learning_rate(_learning_rate) {
            do_not_automatically_include = true;
        }
		
		// ----- PUBLIC METHODS -----
        // select one neuron to track by its index
        void activate_for(size_t neuronIdx) override {
            neuron_mask.emplace_back(static_cast<size_t>(neuronIdx));
        }
        
        // select multiple neurons to track by passing a vector of indices
        void activate_for(std::vector<size_t> neuronIdx) override {
            neuron_mask.insert(neuron_mask.end(), neuronIdx.begin(), neuronIdx.end());
        }
        
        virtual void learn(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            // error handling
            if (time_constant == postsynapticNeuron->get_membrane_time_constant()) {
                throw std::logic_error("the myelin plasticity time constant cannot be equal to the neuron's membrane time constant");
            }
            
            if (network->get_verbose() > 1) {
                std::cout << "New learning epoch at t=" << timestamp << std::endl;
            }
            
            std::vector<float> time_differences;
            std::vector<Synapse*> accepted_synapses;
            
            // saving relevant synapses and their spike times
            for (auto& input: postsynapticNeuron->get_dendritic_tree()) {
                if (input->get_type() == synapse_type::excitatory) {
                    // easy access to the input neuron
                    auto& inputNeuron = network->get_neurons()[input->get_presynaptic_neuron_id()];
                    
                    // arrival time of the input spike
                    double spike_arrival_time = input->get_previous_input_time();
                    
                    // taking the input neurons that were active
                    if (inputNeuron->get_trace() > 0) {
                        accepted_synapses.emplace_back(input);
                        
                        // calculating the time difference
                        float time_difference = static_cast<float>(postsynapticNeuron->get_previous_input_time() - spike_arrival_time);
                        
                        // saving information for the corresponding logger
                        time_differences.emplace_back(time_difference);
                        
                        // change delay according to the time difference
                        float delta_delay = learning_rate * (1/(time_constant - postsynapticNeuron->get_membrane_time_constant())) * postsynapticNeuron->get_current() * (std::exp(-time_difference/time_constant) - std::exp(-time_difference/postsynapticNeuron->get_membrane_time_constant()));
                        
                        input->increment_delay(delta_delay);

                        if (network->get_verbose() > 1) {
                            std::cout << " inside learning window " << spike_arrival_time << " " << input->get_presynaptic_neuron_id() << " " << input->get_postsynaptic_neuron_id() << " time difference: " << time_difference << " delay change: " << delta_delay << " delay: " << input->get_delay() << " trace " << inputNeuron->get_trace() << " threshold " << inputNeuron->get_threshold() << " current: " << postsynapticNeuron->get_current() << " previous input time: "<< postsynapticNeuron->get_previous_input_time() << std::endl;
                        }
                    }
                    
                    // resetting trace for the input neuron
                    inputNeuron->set_trace(0);
                }
            }
        
            // saving into the neuron's logger if the logger exists
            for (auto& addon: postsynapticNeuron->get_relevant_addons()) {
                if (MyelinPlasticityLogger* myelinLogger = dynamic_cast<MyelinPlasticityLogger*>(addon)) {
                    dynamic_cast<MyelinPlasticityLogger*>(addon)->myelin_plasticity_event(timestamp, postsynapticNeuron, network, time_differences, accepted_synapses);
                }
            }
        }
        
	protected:
	
		// ----- LEARNING RULE PARAMETERS -----
        int              time_constant;
        float            learning_rate;
	};
}
