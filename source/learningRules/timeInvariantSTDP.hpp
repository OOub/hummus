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

#include "../globalLearningRuleHandler.hpp"
#include "../neurons/input.hpp"
#include "../neurons/LIF.hpp"

namespace hummus {
	class Neuron;
	
	class TimeInvariantSTDP : public GlobalLearningRuleHandler {
        
	public:
		// ----- CONSTRUCTOR -----
        TimeInvariantSTDP(float _alpha_plus=0.2, float _alpha_minus=-0.025, float _beta_plus=3, float _beta_minus=0, float _leak_time_constant=1, float _leak_scaling_factor=1, float _leak_lower_bound = 0.1, float _leak_upper_bound = 2) :
                alpha_plus(_alpha_plus),
                alpha_minus(_alpha_minus),
                beta_plus(_beta_plus),
                beta_minus(_beta_minus),
                leak_scaling_factor(_leak_scaling_factor),
                leak_time_constant(_leak_time_constant),
                leak_lower_bound(_leak_lower_bound),
                leak_upper_bound(_leak_upper_bound) {}
        
		// ----- PUBLIC METHODS -----
        virtual void onStart(Network* network) override{
            // error handling
            for (auto& n: network->getNeurons()) {
                for (auto& rule: n->getLearningRules()) {
                    if (rule == this) {
                        n->addLearningInfo(std::pair<int, std::vector<float>>(2, {alpha_plus, alpha_minus, beta_plus, beta_minus}));
                        if (n->getLayerID() == 0) {
                            throw std::logic_error("the STDP learning rule has to be on a postsynaptic layer");
                        }
                    }
                }
            }
        }
        
		virtual void learn(double timestamp, synapse* a, Network* network) override {
            for (auto& preSynapse: a->postNeuron->getPreSynapses()) {
                // ignoring inhibitory synapses and ignoring synapses that are outside the [0,1] range
                if (preSynapse->weight >= 0 && preSynapse->weight <= 1) {
                                        
                    // Long term potentiation for all presynaptic neurons that spiked
                    if (timestamp >= preSynapse->preNeuron->getPreviousSpikeTime() && preSynapse->preNeuron->getPreviousSpikeTime() > a->postNeuron->getPreviousSpikeTime()) {
                        // positive weight change
                        float delta_weight = (alpha_plus * std::exp(- beta_plus * preSynapse->weight)) * preSynapse->weight * (1 - preSynapse->weight);
                        preSynapse->weight += delta_weight;
                        
                        if (network->getVerbose() >= 1) {
                            std::cout << "LTP weight change " << delta_weight << " weight " << preSynapse->weight << std::endl;
                        }
                        
                        // calculating leak adaptation
                        float previousLeak = preSynapse->postNeuron->getAdaptation();
						float leakAdaptation = previousLeak + ( - leak_scaling_factor * std::exp( - leak_time_constant * (delta_weight * delta_weight * 10)) + leak_scaling_factor);
						
						// adding hard constrains
						if (leakAdaptation < leak_lower_bound) {
							leakAdaptation = leak_lower_bound;
						}
						if (leakAdaptation > leak_upper_bound) {
							leakAdaptation = leak_upper_bound;
						}
						
						preSynapse->postNeuron->setAdaptation(leakAdaptation);
						if (network->getVerbose() >= 1) {
							std::cout << "LTP leak adaptation " << previousLeak << " " << leakAdaptation << std::endl;
						}
                        
                    // Long term depression for all presynaptic neurons that didn't spike
                    } else {
                        
                        // negative weight change
                        float delta_weight = (alpha_minus * std::exp(- beta_minus * (1 - preSynapse->weight))) * preSynapse->weight * (1 - preSynapse->weight);
                        preSynapse->weight += delta_weight;
                        
                        if (network->getVerbose() >= 1) {
                            std::cout << "LTD weight change " << delta_weight << " weight " << preSynapse->weight << std::endl;
                        }
                        
						// calculating leak adaptation
                        float previousLeak = preSynapse->postNeuron->getAdaptation();
						float leakAdaptation = previousLeak - (- leak_scaling_factor * std::exp( - leak_time_constant * (delta_weight * delta_weight * 10)) + leak_scaling_factor);
						
						// adding hard constrains
						if (leakAdaptation < leak_lower_bound) {
							leakAdaptation = leak_lower_bound;
						}
						if (leakAdaptation > leak_upper_bound) {
							leakAdaptation = leak_upper_bound;
						}
						
						preSynapse->postNeuron->setAdaptation(leakAdaptation);
						if (network->getVerbose() >= 1) {
							std::cout << "LTD leak adaptation " << previousLeak << " " << leakAdaptation << std::endl;
						}
						
                    }
                } else if (preSynapse->weight > 1) {
                    if (network->getVerbose() >= 1) {
                        std::cout << "a synapse has a weight higher than 1, this particular learning rule requires weights to fall within the [0,1] range. The synapse was ignored and will not learn" << std::endl;
                    }
                }
            }
		}
		
	protected:
	
		// ----- LEARNING RULE PARAMETERS -----
        float alpha_plus;
        float alpha_minus;
        float beta_plus;
        float beta_minus;
        float leak_scaling_factor;
        float leak_time_constant;
        float leak_lower_bound;
		float leak_upper_bound;
	};
}
