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

#include "../globalLearningRuleHandler.hpp"
#include "../neurons/inputNeuron.hpp"
#include "../neurons/LIF.hpp"

namespace hummus {
	class Neuron;
	
	class STDP : public GlobalLearningRuleHandler {
        
	public:
		// ----- CONSTRUCTOR -----
		STDP(float _A_plus=1, float _A_minus=1, float _tau_plus=20, float _tau_minus=20) :
                A_plus(_A_plus),
                A_minus(_A_minus),
                tau_plus(_tau_plus),
                tau_minus(_tau_minus) {}
		
		// ----- PUBLIC METHODS -----
		virtual void onStart(Network* network) override  {
			for (auto& n: network->getNeurons()) {
				for (auto& rule: n->getLearningRuleHandler()) {
					if (rule == this) {
						if (n->getLayerID() > 0) {
                            postLayer = n->getLayerID();
                            
                            // making sure we don't add learning on a parallel layer
                            for (auto& preAxon: n->getPreAxons()) {
                                if (preAxon->preNeuron->getLayerID() < preAxon->postNeuron->getLayerID()) {
                                    // finding the closest presynaptic layer without overly relying on layerIDs
                                    preLayer = std::max(preAxon->preNeuron->getLayerID(), preLayer);
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
		
		virtual void learn(double timestamp, axon* a, Network* network) override {
            // LTD whenever a neuron from the presynaptic layer spikes
            if (a->postNeuron->getLayerID() == preLayer) {
                for (auto& postAxon: a->postNeuron->getPostAxons()) {
                    // if a postNeuron fired, the deltaT (preTime - postTime) should be positive
                    if (postAxon->postNeuron->getEligibilityTrace() > 0.1) {
                        float postTrace = - (timestamp - postAxon->postNeuron->getPreviousSpikeTime())/tau_minus * A_minus*std::exp(-(timestamp - postAxon->postNeuron->getPreviousSpikeTime())/tau_minus);
                        
                        if (postAxon->weight > 0) {
                            postAxon->weight += postTrace*(1/postAxon->postNeuron->getMembraneResistance());

                            if (postAxon->weight < 0) {
                                postAxon->weight = 0;
                            }
                        }
                        postAxon->postNeuron->setEligibilityTrace(0);
                    }
                }
            }
			
			// LTP whenever a neuron from the postsynaptic layer spikes
			else if (a->postNeuron->getLayerID() == postLayer) {
				for (auto& preAxon: a->postNeuron->getPreAxons()) {
					// if a preNeuron already fired, the deltaT (preTime - postTime) should be negative
					if (preAxon->preNeuron->getEligibilityTrace() > 0.1) {
						float preTrace = -(preAxon->preNeuron->getPreviousSpikeTime() - timestamp)/tau_plus * A_plus*std::exp((preAxon->preNeuron->getPreviousSpikeTime() - timestamp)/tau_plus);

                        if (preAxon->weight < 1/preAxon->preNeuron->getMembraneResistance()) {
                            preAxon->weight += preTrace*(1/preAxon->preNeuron->getMembraneResistance());

                            if (preAxon->weight > 1/preAxon->preNeuron->getMembraneResistance()) {
                                preAxon->weight = 1/preAxon->preNeuron->getMembraneResistance();
                            }
                        }
                        preAxon->preNeuron->setEligibilityTrace(0);
					}
				}
			}
		}
		
	protected:
	
		// ----- LEARNING RULE PARAMETERS -----
		int16_t preLayer;
		int16_t postLayer;
		float   A_plus;
		float   A_minus;
		float   tau_plus;
		float   tau_minus;
	};
}
