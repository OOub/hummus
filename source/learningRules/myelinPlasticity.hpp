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
#include "../neurons/LIF.hpp"
#include "../addons/myelinPlasticityLogger.hpp"

namespace hummus {
    class Synapse;
	class Neuron;
	
	class MyelinPlasticity : public Addon {
        
	public:
		// ----- CONSTRUCTOR -----
        MyelinPlasticity(float _learning_window_sigma=10, float _learning_rate=1) :
                learning_window_sigma(_learning_window_sigma),
                learning_rate(_learning_rate) {}
		
		// ----- PUBLIC METHODS -----
        // select one neuron to track by its index
        void activate_for(size_t neuronIdx) override {
            neuron_mask.push_back(static_cast<size_t>(neuronIdx));
        }
        
        // select multiple neurons to track by passing a vector of indices
        void activate_for(std::vector<size_t> neuronIdx) override {
            neuron_mask.insert(neuron_mask.end(), neuronIdx.begin(), neuronIdx.end());
        }
        
        virtual void learn(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            if (network->getVerbose() >= 1) {
                std::cout << "New learning epoch at t=" << timestamp << std::endl;
            }
            
            std::vector<double> input_times;
            std::vector<Synapse*> input_synapses;
            
            // forcing the neuron to be a LIF
            LIF* n = dynamic_cast<LIF*>(postsynapticNeuron);
            
            // weight normaliser
            float weight_normaliser = 0;
            
            /// POTENTIATION ON THE WINNER NEURON
            
            // saving relevant synapses and their spike times
            for (auto& input: n->getDendriticTree()) {
                double input_time = input->getPreviousInputTime() + input->getDelay();
                float gaussian_window = gaussian_distribution(input_time, timestamp, learning_window_sigma);
                
                // ignoring inhibitory synapses
                if (input->getWeight() > 0) {
                    // taking the synapses that were active before the current timestamp within a specific learning window
                    if (input->getPreviousInputTime() <= timestamp && gaussian_window >= 0.01) {
                        input_times.push_back(input_time);
                        input_synapses.push_back(input);
                        
                        // calculating the time difference
                        double time_difference = timestamp - input_time;
                        float delta_delay = 0;
                        
                        // change delay according to the time difference
                        if (time_difference > 0) {
                            delta_delay = learning_rate * (1/(n->getDecayCurrent()-n->getDecayPotential())) * n->getCurrent() * (std::exp(-time_difference/n->getDecayCurrent()) - std::exp(-time_difference/n->getDecayPotential()));
                            input->setDelay(delta_delay);
                        } else if (time_difference < 0) {
                            delta_delay = - learning_rate * (1/(n->getDecayCurrent()-n->getDecayPotential())) * n->getCurrent() * (std::exp(time_difference/n->getDecayCurrent()) - std::exp(time_difference/n->getDecayPotential()));
                            input->setDelay(delta_delay);
                        }
                        
                        // increasing weights depending on activity, according to a gaussian on the time difference
                        float delta_weight = learning_rate * gaussian_distribution(time_difference - delta_delay, 0, learning_window_sigma);
                        input->setWeight(delta_weight);
                        
                        if (network->getVerbose() >= 1) {
                            std::cout << timestamp << " " << input->getPresynapticNeuronID() << " " << input->getPostsynapticNeuronID() << " time difference: " << time_difference << " delay change: " << delta_delay << " delay: " << input->getDelay() << " synaptic efficacy: " << input->getSynapticEfficacy() << std::endl;
                        }
                        
                        // decrease the synaptic efficacy as the delays converge
                        input->setSynapticEfficacy(-std::exp(-time_difference * time_difference)+1, false);
                    }
                    
                    // calculating weight normaliser
                    weight_normaliser += input->getWeight();
                }
            }
            
            // normalising synaptic weights
            for (auto& input: n->getDendriticTree()) {
                if (input->getWeight() > 0) {
                    input->setWeight(input->getWeight() / weight_normaliser, false);
                    std::cout << input->getPresynapticNeuronID() << "->" << input->getPostsynapticNeuronID() << " weight: " << input->getWeight() << std::endl;
                }
            }
            
            /// DEPRESSION ON THE LOSER NEURONS TO FURTHER SPECIALISE THE PATTERN
            
        }
        
        inline float gaussian_distribution(float x, float mu, float sigma) {
            return 12.533 * std::exp(- 0.5 * std::pow((x - mu)/sigma, 2)) / (sigma * std::sqrt(2 * M_PI));
        }
        
	protected:
	
		// ----- LEARNING RULE PARAMETERS -----
        float            learning_window_sigma;
        float            learning_rate;
	};
}
