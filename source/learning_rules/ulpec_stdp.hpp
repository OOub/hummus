/*
 * ulpec_stdp.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 22/10/2019
 *
 * Information: Simplified STDP learning rule that is compatible with the ULPEC demonstrator
 */

#pragma once

#include "../addon.hpp"

namespace hummus {
	class Neuron;
	
	class ULPEC_STDP : public Addon {
        
	public:
		// ----- CONSTRUCTOR -----
		ULPEC_STDP(float _A_pot=0.1, float _A_dep=-0.1, float _thres_pot=-1.2, float _thres_dep=1.2, float _G_max=1e-6, float _G_min=1e-8) :
                A_pot(_A_pot),
                A_dep(_A_dep),
                thres_pot(_thres_pot),
                thres_dep(_thres_dep),
                G_max(_G_max),
                G_min(_G_min) {
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
            
            // potentiation            
            float G_0 = s->get_weight();
            if (s->get_synaptic_potential() <= thres_pot) {
                if (network->get_verbose() > 1) {
                    std::cout << " LTP" << std::endl;
                }
                float delta_G = A_pot * (G_max - G_0);
                s->set_weight(G_0+delta_G);
                
            // depression
            } else if (s->get_synaptic_potential() >= thres_dep) {
                if (network->get_verbose() > 1) {
                    std::cout << "LTD" << std::endl;
                }
                float delta_G = A_dep * (G_0 - G_min);
                s->set_weight(G_0+delta_G);
            }
		}
		
	protected:
	
		// ----- LEARNING RULE PARAMETERS -----
		float                A_pot;     // potentiation learning rate
		float                A_dep;     // depression learning rate
		float                thres_pot; // voltage threshold to start potentiation
		float                thres_dep; // voltage threshold to start depression
        float                G_max;     // maximum conductance value by the memristors (conductance = weight)
        float                G_min;     // minimum conductance value by the memristors (conductance = weight)
	};
}
