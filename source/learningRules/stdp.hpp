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

#include "../globalLearningRuleHandler.hpp"
#include "../neurons/input.hpp"
#include "../neurons/LIF.hpp"

namespace hummus {
	class Neuron;
	
	class STDP : public GlobalLearningRuleHandler {
        
	public:
		// ----- CONSTRUCTOR -----
		STDP(float _A_plus=0.1, float _A_minus=0.04, float _tau_plus=20, float _tau_minus=40) :
                A_plus(_A_plus),
                A_minus(_A_minus),
                tau_plus(_tau_plus),
                preLayer(-1),
                tau_minus(_tau_minus) {}
		
		// ----- PUBLIC METHODS -----
		virtual void onStart(Network* network) override  {
			for (auto& n: network->getNeurons()) {
				for (auto& rule: n->getLearningRules()) {
					if (rule == this) {
                        n->addLearningInfo(std::pair<int, std::vector<float>>(1, {A_plus, A_minus, tau_plus, tau_minus}));
						if (n->getLayerID() > 0) {
                            postLayer = n->getLayerID();
                            
                            // making sure we don't add learning on a parallel layer
                            for (auto& preSynapse: n->getPreSynapses()) {
                                if (preSynapse->preNeuron->getLayerID() < preSynapse->postNeuron->getLayerID()) {
                                    // finding the closest presynaptic layer without overly relying on layerIDs
                                    preLayer = std::max(preSynapse->preNeuron->getLayerID(), preLayer);
                                }
                            }
						} else {
							throw std::logic_error("the STDP learning rule has to be on a postsynaptic layer");
						}
					}
				}
			}
			
			for (auto& n: network->getLayers()[preLayer].neurons) {
                network->getNeurons()[n]->addLearningRule(this);
			}
		}
		
		virtual void learn(double timestamp, synapse* a, Network* network) override {
            // LTD whenever a neuron from the presynaptic layer spikes
            if (a->postNeuron->getLayerID() == preLayer) {
                for (auto& postSynapse: a->postNeuron->getPostSynapses()) {
                    // if a postNeuron fired, the deltaT (preTime - postTime) should be positive
                    // ignoring inhibitory synapses
                    if (postSynapse->weight >=0 && postSynapse->postNeuron->getEligibilityTrace() > 0.1) {
                        float postTrace = (- A_minus * std::exp(-(timestamp - postSynapse->postNeuron->getPreviousSpikeTime())/tau_minus)) * (1 - postSynapse->weight);
                        
                        postSynapse->weight += postTrace;
                        
                        if (network->getVerbose() >= 1) {
                            std::cout << "LTD weight change " << postTrace << std::endl;
                        }
                        
                        if (postSynapse->weight < 0) {
                            postSynapse->weight = 0;
                        }
                    }
                }
            }
			
			// LTP whenever a neuron from the postsynaptic layer spikes
			else if (a->postNeuron->getLayerID() == postLayer) {
				for (auto& preSynapse: a->postNeuron->getPreSynapses()) {
					// if a preNeuron already fired, the deltaT (preTime - postTime) should be negative
                    // ignoring inhibitory synapses
					if (preSynapse->weight >= 0 && preSynapse->preNeuron->getEligibilityTrace() > 0.1) {
						float preTrace = (A_plus * std::exp((preSynapse->preNeuron->getPreviousSpikeTime() - timestamp)/tau_plus)) * (1 - preSynapse->weight);
                        
                        preSynapse->weight += preTrace;
                        
                        if (network->getVerbose() >= 1) {
                            std::cout << "LTP weight change " << preTrace << std::endl;
                        }
					}
				}
			}
		}
		
	protected:
	
		// ----- LEARNING RULE PARAMETERS -----
		int     preLayer;
		int     postLayer;
		float   A_plus;
		float   A_minus;
		float   tau_plus;
		float   tau_minus;
	};
}
