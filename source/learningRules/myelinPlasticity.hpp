/*
 * myelinPlasticity.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: The MyelinPlasticity learning rule compatible only with leaky integrate-and-fire neurons.
 *
 * LEARNING RULE TYPE 0 (in JSON SAVE FILE)
 */

#pragma once

#include <stdexcept>
#define _USE_MATH_DEFINES

#include "../addon.hpp"
#include "../addons/myelinPlasticityLogger.hpp"

namespace hummus {
    class Synapse;
	class Neuron;
	
	class MyelinPlasticity : public Addon {
        
	public:
		// ----- CONSTRUCTOR -----
        MyelinPlasticity(int _time_constant=10, int _learning_window=20, float _learning_rate=1, float _alpha_plus=0.2, float _alpha_minus=-0.08, float _beta_plus=0.1, float _beta_minus=0) :
                time_constant(_time_constant),
                learning_window(_learning_window),
                learning_rate(_learning_rate),
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
        
        virtual void learn(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            // error handling
            if (time_constant == postsynapticNeuron->getMembraneTimeConstant()) {
                throw std::logic_error("the myelin plasticity time constant cannot be equal to the neuron's membrane time constant");
            }
            
            if (network->getVerbose() >= 1) {
                std::cout << "New learning epoch at t=" << timestamp << std::endl;
            }
            
            std::vector<double> time_differences;
            std::vector<Synapse*> active_synapses;
            
            // weight normaliser
            float weight_normaliser = 0;
            
            // saving relevant synapses and their spike times
            for (auto& input: postsynapticNeuron->getDendriticTree()) {
                if (input->getType() == synapseType::excitatory) {
                    // easy access to the input neuron
                    auto& inputNeuron = network->getNeurons()[input->getPresynapticNeuronID()];
                    
                    // arrival time of the input spike
                    double spike_arrival_time = input->getPreviousInputTime();
                    
                    // learning window
                    float gaussian_window = gaussian_distribution(spike_arrival_time, timestamp, learning_window);
                    
                    // taking the input neurons that were active within a gaussian learning window
                    if (gaussian_window >= 0.01 && inputNeuron->getTrace() > 0) {
                        active_synapses.emplace_back(input);
                        
                        // calculating the time difference
                        double time_difference = postsynapticNeuron->getPreviousInputTime() - spike_arrival_time;
                        
                        // saving information for the corresponding logger
                        time_differences.emplace_back(time_difference);
                        
                        // change delay according to the time difference
                        float delta_delay = 0;
                        delta_delay = learning_rate * (1/(time_constant - postsynapticNeuron->getMembraneTimeConstant())) * postsynapticNeuron->getCurrent() * (std::exp(-time_difference/time_constant) - std::exp(-time_difference/postsynapticNeuron->getMembraneTimeConstant()));
                        input->setDelay(delta_delay);
                        
                        // long-term potentiation on weights
                        float delta_weight = (alpha_plus * std::exp(- time_difference * beta_plus * input->getWeight())) * input->getWeight() * (1 - input->getWeight());
                        input->setWeight(delta_weight);
                        
                        // calculating weight normaliser
                        weight_normaliser += input->getWeight();
                        
                        if (network->getVerbose() >= 1) {
                            std::cout << spike_arrival_time << " " << input->getPresynapticNeuronID() << " " << input->getPostsynapticNeuronID() << " time difference: " << time_difference << " delay change: " << delta_delay << " delay: " << input->getDelay() << " weight change: " << delta_weight << " weight " << input->getWeight() << std::endl;
                        }
                        
                    // taking the input neurons that are outside the learning window or that didn't spike
                    } else {
                        // long-term depression on weights
                        float delta_weight = (alpha_minus * std::exp(- beta_minus * (1 - input->getWeight()))) * input->getWeight() * (1 - input->getWeight());
                        input->setWeight(delta_weight);
                        
                        if (network->getVerbose() >= 1) {
                            std::cout << input->getPresynapticNeuronID() << " " << input->getPostsynapticNeuronID() << " weight change: " << delta_weight << " weight " << input->getWeight() << std::endl;
                        }
                    }
                }
            }
            
            // normalising synaptic weights on active synapses
            for (auto& active_synapse: active_synapses) {
                active_synapse->setWeight(active_synapse->getWeight()/ weight_normaliser, false);
            }
            
            // printing the weights
            if (network->getVerbose() >= 1) {
                for (auto& input: postsynapticNeuron->getDendriticTree()) {
                    if (input->getType() == synapseType::excitatory) {
                        std::cout << input->getPresynapticNeuronID() << "->" << input->getPostsynapticNeuronID() << " weight: " << input->getWeight() << std::endl;
                    }
                }
            }
            
            // saving into the neuron's logger if the logger exists
            for (auto& addon: postsynapticNeuron->getRelevantAddons()) {
                if (MyelinPlasticityLogger* myelinLogger = dynamic_cast<MyelinPlasticityLogger*>(addon)) {
                    dynamic_cast<MyelinPlasticityLogger*>(addon)->myelinPlasticityEvent(timestamp, postsynapticNeuron, network, time_differences, active_synapses);
                }
            }
        }
        
        // gaussian distribution with an amplitude peak of 1
        inline float gaussian_distribution(float x, float mu, float sigma) {
            return 12.533 * sigma / 5 * std::exp(- 0.5 * std::pow((x - mu)/sigma, 2)) / (sigma * std::sqrt(2 * M_PI));
        }
        
	protected:
	
		// ----- LEARNING RULE PARAMETERS -----
        int              time_constant;
        int              learning_window;
        float            learning_rate;
        float            alpha_plus;
        float            alpha_minus;
        float            beta_plus;
        float            beta_minus;
	};
}

/*for (auto& dendrite: targetNeuron->getDendriticTree()) {
    // ignoring inhibitory synapses and ignoring synapses that are outside the [0,1] range
    if (dendrite->getType() == synapseType::excitatory && dendrite->getWeight() <= 1) {
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
*/
