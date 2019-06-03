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

#include "../neurons/decisionMaking.hpp"
#include "../addon.hpp"

namespace hummus {
	struct reinforcementLayers {
		int postLayer;
		int preLayer;
	};
	
	class Neuron;
	
	class RewardModulatedSTDP : public Addon {
        
	public:
		// ----- CONSTRUCTOR -----
		RewardModulatedSTDP(float _Ar_plus=1, float _Ar_minus=-1, float _Ap_plus=1, float _Ap_minus=-1, float _leak_time_constant=0.1, float _leak_scaling_factor=1, float _leak_lower_bound = 0.1, float _leak_upper_bound = 2) :
                Ar_plus(_Ar_plus),
                Ar_minus(_Ar_minus),
                Ap_plus(_Ap_plus),
                Ap_minus(_Ap_minus),
                leak_time_constant(_leak_time_constant),
                leak_scaling_factor(_leak_scaling_factor),
                leak_lower_bound(_leak_lower_bound),
                leak_upper_bound(_leak_upper_bound) {
					
			if (Ar_plus <= 0 || Ap_plus <= 0) {
				throw std::logic_error("Ar_plus and Ap_plus need to be positive");
			}
			
			if (Ar_minus >= 0 || Ap_minus >= 0) {
				throw std::logic_error("Ar_minus and Ap_minus need to be negative");
			}
		}
		
		// ----- PUBLIC METHODS -----
		virtual void initialise(Network* network) override {
            for (auto& l: network->getLayers()) {
                for (auto& addon: network->getNeurons()[l.neurons[0]]->getRelevantAddons()) {
                    if (addon == this) {
                        network->getNeurons()[l.neurons[0]]->addLearningInfo(std::pair<int, std::vector<float>>(3, {Ar_plus, Ar_minus, Ap_plus, Ap_minus, leak_time_constant, leak_scaling_factor, leak_lower_bound, leak_upper_bound}));
                        int presynapticLayer = -1;
                        // making sure we don't add learning on a parallel layer
                        for (auto& preSynapse: network->getNeurons()[l.neurons[0]]->getPreSynapses()) {
                            // finding the closest presynaptic layer without overly relying on layerIDs
                            if (preSynapse->preNeuron && preSynapse->preNeuron->getLayerID() < preSynapse->postNeuron->getLayerID()) {
                                presynapticLayer = std::max(preSynapse->preNeuron->getLayerID(), presynapticLayer);
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
		
		virtual void learn(double timestamp, synapse* a, Network* network) override {
		
            if (DecisionMaking* n = dynamic_cast<DecisionMaking*>(a->postNeuron)) {
            	
				// reward and punishement signal from the decision-making layer
				int alpha = 0;
				int beta = 0;
				if (dynamic_cast<DecisionMaking*>(a->postNeuron)->getClassLabel() == network->getCurrentLabel()) {
					alpha = 1;
				} else {
					beta = 1;
				}
				
				// propagating the error signal to every layer using the R-STDP learning rule
				for (auto& layer: rl) {
					// if preTime - postTime is positive
					for (auto& n: network->getLayers()[layer.preLayer].neurons) {
                        if (network->getNeurons()[n]->getEligibilityTrace() > 0.1) {
                            for (auto& postSynapse: network->getNeurons()[n]->getPostSynapses()) {
                                // ignoring inhibitory synapses
                                if (postSynapse->weight >= 0 && postSynapse->weight <= 1 && postSynapse->postNeuron->getEligibilityTrace() > 0.1) {
                                    double delta = alpha*Ar_minus+beta*Ap_plus;
                                    
                                    if (network->getVerbose() >= 1) {
                                        std::cout << "weight change " << delta * postSynapse->weight * (1 - postSynapse->weight) << std::endl;
                                    }
                                    
                                    postSynapse->weight += delta * postSynapse->weight * (1 - postSynapse->weight);
									
                                    // calculating leak adaptation
									float previousLeak = a->postNeuron->getAdaptation();
									float leakAdaptation = previousLeak;
									if (delta < 0) {
										leakAdaptation = previousLeak - (- leak_scaling_factor * std::exp( - leak_time_constant * (delta * delta)) + leak_scaling_factor);
									} else if (delta > 0) {
										leakAdaptation = previousLeak + (- leak_scaling_factor * std::exp( - leak_time_constant * (delta * delta)) + leak_scaling_factor);
									}
										
									
									// adding hard constrains
									if (leakAdaptation < leak_lower_bound) {
										leakAdaptation = leak_lower_bound;
									}
									if (leakAdaptation > leak_upper_bound) {
										leakAdaptation = leak_upper_bound;
									}
									
									a->postNeuron->setAdaptation(leakAdaptation);
									if (network->getVerbose() >= 1) {
										std::cout << "leak adaptation " << previousLeak << " " << leakAdaptation << std::endl;
									}
						
                                } else if (postSynapse->weight > 1) {
                                    if (network->getVerbose() >= 1) {
                                        std::cout << "a synapse has a weight higher than 1, this particular learning rule requires weights to fall within the [0,1] range. The synapse was ignored and will not learn" << std::endl;
                                    }
                                }
                            }
                        }
					}

					// if preTime - postTime is negative
					for (auto& n: network->getLayers()[layer.postLayer].neurons) {
                        if (network->getNeurons()[n]->getEligibilityTrace() > 0.1) {
                            for (auto& preSynapse: network->getNeurons()[n]->getPreSynapses()) {
                                // ignoring inhibitory synapses
                                if (preSynapse->weight >= 0 && preSynapse->weight <= 1 && preSynapse->preNeuron->getEligibilityTrace() > 0.1) {
                                    double delta = alpha*Ar_plus+beta*Ap_minus;
                                    
                                    if (network->getVerbose() >= 1) {
                                        std::cout << "weight change " << delta * preSynapse->weight * (1 - preSynapse->weight) << std::endl;
                                    }
                                    
                                    preSynapse->weight += delta * preSynapse->weight * (1 - preSynapse->weight);

									// calculating leak adaptation
									float previousLeak = a->postNeuron->getAdaptation();
									float leakAdaptation = 0;
									if (delta < 0) {
										leakAdaptation = previousLeak - (- leak_scaling_factor * std::exp( - leak_time_constant * (delta * delta)) + leak_scaling_factor);
									} else if (delta > 0) {
										leakAdaptation = previousLeak + (- leak_scaling_factor * std::exp( - leak_time_constant * (delta * delta)) + leak_scaling_factor);
									}
									
									// adding hard constrains
									if (leakAdaptation < leak_lower_bound) {
										leakAdaptation = leak_lower_bound;
									}
									if (leakAdaptation > leak_upper_bound) {
										leakAdaptation = leak_upper_bound;
									}
									
									a->postNeuron->setAdaptation(leakAdaptation);
									if (network->getVerbose() >= 1) {
										std::cout << "leak adaptation " << previousLeak << " " << leakAdaptation << std::endl;
									}
									
                                } else if (preSynapse->weight > 1) {
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
        float                            leak_scaling_factor;
        float                            leak_time_constant;
        float                            leak_lower_bound;
		float                            leak_upper_bound;
	};
}
