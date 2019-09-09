/*
 * stdp.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 23/01/2019
 *
 * Information: The stdp learning rule has to be on a postsynaptic layer because it automatically detects the presynaptic layer.
 * Adapted From: Galluppi, F., Lagorce, X., Stromatias, E., Pfeiffer, M., Plana, L. A., Furber, S. B., & Benosman, R. B. (2015). A framework for plasticity implementation on the SpiNNaker neural architecture. Frontiers in Neuroscience, 8. doi:10.3389/fnins.2014.00429
 */

#pragma once

#include "../addon.hpp"

namespace hummus {
	class Neuron;
	
	class STDP : public Addon {
        
	public:
		// ----- CONSTRUCTOR -----
		STDP(float _A_plus=1, float _A_minus=0.4, float _tau_plus=20, float _tau_minus=40) :
                A_plus(_A_plus),
                A_minus(_A_minus),
                tau_plus(_tau_plus),
                pre_layer(-1),
                tau_minus(_tau_minus) {
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
        
		virtual void on_start(Network* network) override {
            if (!neuron_mask.empty()) {
                for (auto& n: network->get_neurons()) {
                    for (auto& addon: n->get_relevant_addons()) {
                        if (addon == this) {
                            if (n->get_layer_id() > 0) {
                                post_layer = n->get_layer_id();
                                // making sure we don't add learning on a parallel layer
                                for (auto& dendrite: n->get_dendritic_tree()) {
                                    auto& d_presynapticNeuron = network->get_neurons()[dendrite->get_presynaptic_neuron_id()];
                                    auto& d_postsynapticNeuron = network->get_neurons()[dendrite->get_postsynaptic_neuron_id()];
                                    if (d_presynapticNeuron->get_layer_id() < d_postsynapticNeuron->get_layer_id()) {
                                        // finding the closest presynaptic layer without overly relying on layerIDs
                                        pre_layer = std::max(d_presynapticNeuron->get_layer_id(), pre_layer);
                                    }
                                }
                            }
                        }
                    }
                }
                
                for (auto& n: network->get_layers()[pre_layer].neurons) {
                    network->get_neurons()[n]->add_relevant_addon(this);
                }
            }
		}
		
		virtual void learn(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            // LTD whenever a neuron from the presynaptic layer spikes
            if (postsynapticNeuron->get_layer_id() == pre_layer) {
                for (auto& axonTerminal: postsynapticNeuron->get_axon_terminals()) {
                    auto& at_postsynapticNeuron = network->get_neurons()[axonTerminal->get_postsynaptic_neuron_id()];
                    
                    // if a postsynapticNeuron fired, the deltaT (presynaptic time - postsynaptic time) should be positive
                    // ignoring inhibitory synapses
                    if (axonTerminal->get_type() == synapseType::excitatory && axonTerminal->get_weight() <= 1 && at_postsynapticNeuron->get_trace() > 0.1) {
                        float postTrace = (- A_minus * std::exp(-(timestamp - at_postsynapticNeuron->get_previous_spike_time())/tau_minus)) * axonTerminal->get_weight() * (1 - axonTerminal->get_weight());
                        
                        axonTerminal->set_weight(postTrace);
                        
                        if (network->get_verbose() >= 1) {
                            std::cout << "LTD weight change " << postTrace << std::endl;
                        }
						
                    } else if (axonTerminal->get_weight() > 1) {
                        if (network->get_verbose() >= 1) {
                            std::cout << "a synapse has a weight higher than 1, this particular learning rule requires weights to fall within the [0,1] range. The synapse was ignored and will not learn" << std::endl;
                        }
                    }
                }
            }
			
			// LTP whenever a neuron from the postsynaptic layer spikes
			else if (postsynapticNeuron->get_layer_id() == post_layer) {
                for (auto& dendrite: postsynapticNeuron->get_dendritic_tree()) {
                    auto& d_presynapticNeuron = network->get_neurons()[dendrite->get_presynaptic_neuron_id()];
					// if a presynapticNeuron already fired, the deltaT (presynaptic time - postsynaptic time) should be negative
                    // ignoring inhibitory synapses
                    if (dendrite->get_type() == synapseType::excitatory && dendrite->get_weight() <= 1 && d_presynapticNeuron->get_trace() > 0.1) {
                        float preTrace = (A_plus * std::exp((d_presynapticNeuron->get_previous_spike_time() - timestamp)/tau_plus)) * dendrite->get_weight() * (1 - dendrite->get_weight());
                        dendrite->set_weight(preTrace);
                        
                        if (network->get_verbose() >= 1) {
                            std::cout << "LTP weight change " << preTrace << std::endl;
                        }
						
                    } else if (dendrite->get_weight() > 1) {
                        if (network->get_verbose() >= 1) {
                            std::cout << "a synapse has a weight higher than 1, this particular learning rule requires weights to fall within the [0,1] range. The synapse was ignored and will not learn" << std::endl;
                        }
                    }
				}
			}
		}
		
	protected:
	
		// ----- LEARNING RULE PARAMETERS -----
		int                  pre_layer;
		int                  post_layer;
		float                A_plus;
		float                A_minus;
		float                tau_plus;
		float                tau_minus;
	};
}
