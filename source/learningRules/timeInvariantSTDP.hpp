/*
 * timeInvariantSTDP.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 08/02/2019
 *
 * Information: The timeInvariantSTDP learning rule works locally on a layer and does not depend on precise timing (the sign of the postsynaptic neuron time - postsynaptic neuron time only matters)
 * Adapted From: Thiele, J. C., Bichler, O., & Dupret, A. (2018). Event-Based, Timescale Invariant Unsupervised Online Deep Learning With STDP. Frontiers in Computational Neuroscience, 12. doi:10.3389/fncom.2018.00046
 *
 * LEARNING RULE TYPE 2 (in JSON SAVE FILE)
 */

#pragma once

#include "../addon.hpp"

namespace hummus {
	class Neuron;
	
	class TimeInvariantSTDP : public Addon {
        
	public:
		// ----- CONSTRUCTOR -----
        TimeInvariantSTDP(float _alpha_plus=0.2, float _alpha_minus=-0.025, float _beta_plus=3, float _beta_minus=0) :
                alpha_plus(_alpha_plus),
                alpha_minus(_alpha_minus),
                beta_plus(_beta_plus),
                beta_minus(_beta_minus) {}
        
		// ----- PUBLIC METHODS -----
        // select one neuron to track by its index
        void activate_for(size_t neuronIdx) override {
            neuron_mask.emplace_back(static_cast<size_t>(neuronIdx));
        }
        
        // select multiple neurons to track by passing a vector of indices
        void activate_for(std::vector<size_t> neuronIdx) override {
            neuron_mask.insert(neuron_mask.end(), neuronIdx.begin(), neuronIdx.end());
        }
        
        virtual void onStart(Network* network) override{
            // error handling
            for (auto& n: network->getNeurons()) {
                for (auto& addon: n->getRelevantAddons()) {
                    if (addon == this && n->getLayerID() == 0) {
                        throw std::logic_error("the STDP learning rule has to be on a postsynaptic layer");
                    }
                }
            }
        }
        
		virtual void learn(double timestamp, Synapse* s, Neuron* targetNeuron, Network* network) override {
            for (auto& dendrite: targetNeuron->getDendriticTree()) {
                // ignoring inhibitory synapses and ignoring synapses that are outside the [0,1] range
                if (dendrite->getWeight() >= 0 && dendrite->getWeight() <= 1) {
                    auto& d_presynapticNeuron = network->getNeurons()[dendrite->getPresynapticNeuronID()];
                    // Long term potentiation for all presynaptic neurons that spiked
                    if (timestamp >= d_presynapticNeuron->getPreviousSpikeTime() && d_presynapticNeuron->getPreviousSpikeTime() > targetNeuron->getPreviousSpikeTime()) {
                        // positive weight change
                        float delta_weight = (alpha_plus * std::exp(- beta_plus * dendrite->getWeight())) * dendrite->getWeight() * (1 - dendrite->getWeight());
                        dendrite->setWeight(delta_weight);
                        
                        if (network->getVerbose() >= 1) {
                            std::cout << "LTP weight change " << delta_weight << " weight " << dendrite->getWeight() << std::endl;
                        }
                    // Long term depression for all presynaptic neurons that didn't spike
                    } else {
                        
                        // negative weight change
                        float delta_weight = (alpha_minus * std::exp(- beta_minus * (1 - dendrite->getWeight()))) * dendrite->getWeight() * (1 - dendrite->getWeight());
                        dendrite->setWeight(delta_weight);
                        
                        if (network->getVerbose() >= 1) {
                            std::cout << "LTD weight change " << delta_weight << " weight " << dendrite->getWeight() << std::endl;
                        }
                    }
                } else if (dendrite->getWeight() > 1) {
                    if (network->getVerbose() >= 1) {
                        std::cout << "a synapse has a weight higher than 1, this particular learning rule requires weights to fall within the [0,1] range. The synapse was ignored and will not learn" << std::endl;
                    }
                }
            }
		}
		
	protected:
	
		// ----- LEARNING RULE PARAMETERS -----
        float                alpha_plus;
        float                alpha_minus;
        float                beta_plus;
        float                beta_minus;
	};
}
