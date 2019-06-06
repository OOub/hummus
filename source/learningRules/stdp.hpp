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
 *
 * LEARNING RULE TYPE 1 (in JSON SAVE FILE)
 */

#pragma once

#include "../addon.hpp"
#include "../neurons/input.hpp"
#include "../neurons/LIF.hpp"

namespace hummus {
	class Neuron;
	
	class STDP : public Addon {
        
	public:
		// ----- CONSTRUCTOR -----
		STDP(float _A_plus=1, float _A_minus=0.4, float _tau_plus=20, float _tau_minus=40) :
                A_plus(_A_plus),
                A_minus(_A_minus),
                tau_plus(_tau_plus),
                preLayer(-1),
                tau_minus(_tau_minus) {}
		
		// ----- PUBLIC METHODS -----
        // select one neuron to track by its index
        void activate_for(size_t neuronIdx) override {
            neuron_mask.push_back(static_cast<size_t>(neuronIdx));
        }
        
        // select multiple neurons to track by passing a vector of indices
        void activate_for(std::vector<size_t> neuronIdx) override {
            neuron_mask.insert(neuron_mask.end(), neuronIdx.begin(), neuronIdx.end());
        }
        
		virtual void onStart(Network* network) override {
			for (auto& n: network->getNeurons()) {
				for (auto& addon: n->getRelevantAddons()) {
					if (addon == this) {
						if (n->getLayerID() > 0) {
                            postLayer = n->getLayerID();
                            // making sure we don't add learning on a parallel layer
                            for (auto& dendrite: n->getDendriticTree()) {
                                auto& d_presynapticNeuron = network->getNeurons()[dendrite->getPresynapticNeuronID()];
                                auto& d_postsynapticNeuron = network->getNeurons()[dendrite->getPostsynapticNeuronID()];
                                if (d_presynapticNeuron->getLayerID() < d_postsynapticNeuron->getLayerID()) {
                                    // finding the closest presynaptic layer without overly relying on layerIDs
                                    preLayer = std::max(d_presynapticNeuron->getLayerID(), preLayer);
                                }
                            }
						} else {
							throw std::logic_error("the STDP learning rule has to be on a postsynaptic layer");
						}
					}
				}
			}
			
			for (auto& n: network->getLayers()[preLayer].neurons) {
                network->getNeurons()[n]->addRelevantAddon(this);
			}
		}
		
		virtual void learn(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            // LTD whenever a neuron from the presynaptic layer spikes
            if (postsynapticNeuron->getLayerID() == preLayer) {
                for (auto& axonTerminal: postsynapticNeuron->getAxonTerminals()) {
                    auto& at_postsynapticNeuron = network->getNeurons()[axonTerminal->getPostsynapticNeuronID()];
                    
                    // if a postsynapticNeuron fired, the deltaT (presynaptic time - postsynaptic time) should be positive
                    // ignoring inhibitory synapses
                    if (axonTerminal->getWeight() >=0 && axonTerminal->getWeight() <= 1 && at_postsynapticNeuron->getEligibilityTrace() > 0.1) {
                        float postTrace = (- A_minus * std::exp(-(timestamp - at_postsynapticNeuron->getPreviousSpikeTime())/tau_minus)) * axonTerminal->getWeight() * (1 - axonTerminal->getWeight());
                        
                        axonTerminal->setWeight(postTrace);
                        
                        if (network->getVerbose() >= 1) {
                            std::cout << "LTD weight change " << postTrace << std::endl;
                        }
						
                    } else if (axonTerminal->getWeight() > 1) {
                        if (network->getVerbose() >= 1) {
                            std::cout << "a synapse has a weight higher than 1, this particular learning rule requires weights to fall within the [0,1] range. The synapse was ignored and will not learn" << std::endl;
                        }
                    }
                }
            }
			
			// LTP whenever a neuron from the postsynaptic layer spikes
			else if (postsynapticNeuron->getLayerID() == postLayer) {
				for (auto& dendrite: postsynapticNeuron->getDendriticTree()) {
                    auto& d_presynapticNeuron = network->getNeurons()[dendrite->getPresynapticNeuronID()];
					// if a presynapticNeuron already fired, the deltaT (presynaptic time - postsynaptic time) should be negative
                    // ignoring inhibitory synapses
					if (dendrite->getWeight() >= 0 && dendrite->getWeight() <= 1 && d_presynapticNeuron->getEligibilityTrace() > 0.1) {
						float preTrace = (A_plus * std::exp((d_presynapticNeuron->getPreviousSpikeTime() - timestamp)/tau_plus)) * dendrite->getWeight() * (1 - dendrite->getWeight());
                        dendrite->setWeight(preTrace);
                        
                        if (network->getVerbose() >= 1) {
                            std::cout << "LTP weight change " << preTrace << std::endl;
                        }
						
                    } else if (dendrite->getWeight() > 1) {
                        if (network->getVerbose() >= 1) {
                            std::cout << "a synapse has a weight higher than 1, this particular learning rule requires weights to fall within the [0,1] range. The synapse was ignored and will not learn" << std::endl;
                        }
                    }
				}
			}
		}
		
	protected:
	
		// ----- LEARNING RULE PARAMETERS -----
		int                  preLayer;
		int                  postLayer;
		float                A_plus;
		float                A_minus;
		float                tau_plus;
		float                tau_minus;
	};
}
