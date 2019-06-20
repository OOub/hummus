/*
 * rewardModulatedSTDP.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: The rewardModulatedSTDP learning rule has to be on a postsynaptic layer because it automatically detects the presynaptic layer.
 * Adapted From: Mozafari, M., Ganjtabesh, M., Nowzari-Dalini, A., Thorpe, S. J., Masquelier T. (2018). Combining STDP and Reward-Modulated STDP in Deep Convolutional Spiking Neural Networks for Digit Recognition. arXiv:1804.00227
 *
 * LEARNING RULE TYPE 3 (in JSON SAVE FILE)
 */

#pragma once

#include <algorithm>

#include "../addon.hpp"
#include "../neurons/decisionMaking.hpp"

namespace hummus {
	struct reinforcementLayers {
		int postLayer;
		int preLayer;
	};
	
	class Neuron;
	
	class RewardModulatedSTDP : public Addon {
        
	public:
		// ----- CONSTRUCTOR -----
		RewardModulatedSTDP(float _Ar_plus=1, float _Ar_minus=-1, float _Ap_plus=1, float _Ap_minus=-1) :
                Ar_plus(_Ar_plus),
                Ar_minus(_Ar_minus),
                Ap_plus(_Ap_plus),
                Ap_minus(_Ap_minus) {
					
			if (Ar_plus <= 0 || Ap_plus <= 0) {
				throw std::logic_error("Ar_plus and Ap_plus need to be positive");
			}
			
			if (Ar_minus >= 0 || Ap_minus >= 0) {
				throw std::logic_error("Ar_minus and Ap_minus need to be negative");
			}
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
        
		virtual void onStart(Network* network) override {
            for (auto& l: network->getLayers()) {
                for (auto& addon: network->getNeurons()[l.neurons[0]]->getRelevantAddons()) {
                    if (addon == this) {
                        int presynapticLayer = -1;
                        // making sure we don't add learning on a parallel layer
                        for (auto& dendrite: network->getNeurons()[l.neurons[0]]->getDendriticTree()) {
                            auto& d_presynapticNeuron = network->getNeurons()[dendrite->getPresynapticNeuronID()];
                            // finding the closest presynaptic layer without overly relying on layerIDs
                            if (dendrite->getPresynapticNeuronID() != -1 && d_presynapticNeuron->getLayerID() < d_presynapticNeuron->getLayerID()) {
                                presynapticLayer = std::max(d_presynapticNeuron->getLayerID(), presynapticLayer);
                            }
                        }
                        
                        if (presynapticLayer != -1) {
                            rl.emplace_back(reinforcementLayers{network->getNeurons()[l.neurons[0]]->getLayerID(), presynapticLayer});
                        } else {
                            throw std::logic_error("the reward-modulated STDP learning rule cannot be on the input layer");
                        }
                    }
                }
                
                // add rstdp to the decision-making layer
                if (DecisionMaking* neuron = dynamic_cast<DecisionMaking*>(network->getNeurons()[l.neurons[0]].get())) {
                    for (auto& n: l.neurons) {
                        auto it = std::find(network->getNeurons()[n].get()->getRelevantAddons().begin(), network->getNeurons()[n].get()->getRelevantAddons().end(), this);
                        if (it == network->getNeurons()[n].get()->getRelevantAddons().end()) {
                            dynamic_cast<DecisionMaking*>(network->getNeurons()[n].get())->addRelevantAddon(this);
                        }
                    }
                }
            }
		}
		
		virtual void learn(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
		
            if (DecisionMaking* n = dynamic_cast<DecisionMaking*>(network->getNeurons()[s->getPostsynapticNeuronID()].get())) {
            	
				// reward and punishement signal from the decision-making layer
				int alpha = 0;
				int beta = 0;
				if (dynamic_cast<DecisionMaking*>(postsynapticNeuron)->getClassLabel() == network->getCurrentLabel()) {
					alpha = 1;
				} else {
					beta = 1;
				}
				
				// propagating the error signal to every layer using the R-STDP learning rule
				for (auto& layer: rl) {
					// if presynaptic time - postsynaptic time is positive
					for (auto& n: network->getLayers()[layer.preLayer].neurons) {
                        if (network->getNeurons()[n]->getTrace() > 0.1) {
                            for (auto& axonTerminal: network->getNeurons()[n]->getAxonTerminals()) {
                                auto& at_postsynapticNeuron = network->getNeurons()[axonTerminal->getPostsynapticNeuronID()];
                                
                                // ignoring inhibitory synapses
                                if (axonTerminal->getWeight() >= 0 && axonTerminal->getWeight() <= 1 && at_postsynapticNeuron->getTrace() > 0.1) {
                                    double delta = alpha*Ar_minus+beta*Ap_plus;
                                    
                                    if (network->getVerbose() >= 1) {
                                        std::cout << "weight change " << delta * axonTerminal->getWeight() * (1 - axonTerminal->getWeight()) << std::endl;
                                    }
                                    
                                    axonTerminal->setWeight(delta * axonTerminal->getWeight() * (1 - axonTerminal->getWeight()));
						
                                } else if (axonTerminal->getWeight() > 1) {
                                    if (network->getVerbose() >= 1) {
                                        std::cout << "a synapse has a weight higher than 1, this particular learning rule requires weights to fall within the [0,1] range. The synapse was ignored and will not learn" << std::endl;
                                    }
                                }
                            }
                        }
					}

					// if presynaptic time - postsynaptic time is positive
					for (auto& n: network->getLayers()[layer.postLayer].neurons) {
                        if (network->getNeurons()[n]->getTrace() > 0.1) {
                            for (auto& dendrite: network->getNeurons()[n]->getDendriticTree()) {
                                auto& d_presynapticNeuron = network->getNeurons()[dendrite->getPresynapticNeuronID()];
                                
                                // ignoring inhibitory synapses
                                if (dendrite->getWeight() >= 0 && dendrite->getWeight() <= 1 && d_presynapticNeuron->getTrace() > 0.1) {
                                    double delta = alpha*Ar_plus+beta*Ap_minus;
                                    
                                    if (network->getVerbose() >= 1) {
                                        std::cout << "weight change " << delta * dendrite->getWeight() * (1 - dendrite->getWeight()) << std::endl;
                                    }
                                    
                                    dendrite->setWeight(delta * dendrite->getWeight() * (1 - dendrite->getWeight()));
									
                                } else if (dendrite->getWeight() > 1) {
                                    if (network->getVerbose() >= 1) {
                                        std::cout << "a synapse has a weight higher than 1, this particular learning rule requires weights to fall within the [0,1] range. The synapse was ignored and will not learn" << std::endl;
                                    }
                                }
                            }
                        }
					}
				}
			}
		}
		
	protected:
	
		// ----- LEARNING RULE PARAMETERS -----
		std::vector<reinforcementLayers> rl;
		float                            Ar_plus;
		float                            Ar_minus;
		float                            Ap_plus;
		float                            Ap_minus;
	};
}
