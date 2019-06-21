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
        MyelinPlasticity(int _delay_learning_window_sigma=20, int _weight_learning_window_sigma=10, float _learning_rate=1) :
                delay_learning_window_sigma(_delay_learning_window_sigma),
                weight_learning_window_sigma(_weight_learning_window_sigma),
                learning_rate(_learning_rate) {}
		
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
            if (network->getVerbose() >= 1) {
                std::cout << "New learning epoch at t=" << timestamp << std::endl;
            }
            
            std::vector<double> time_differences;
            std::vector<std::vector<int>> plastic_coordinates(4);
            
            // weight normaliser
            float weight_normaliser = 0;
            
            /// CONDUCTION DELAY CONVERGENCE ON THE WINNER NEURON
            
            // saving relevant synapses and their spike times
            for (auto& input: postsynapticNeuron->getDendriticTree()) {
                if (input->getWeight() > 0) {
                    double spike_arrival_time = input->getPreviousInputTime();
                    
                    // learning window
                    float gaussian_window = gaussian_distribution(spike_arrival_time, timestamp, delay_learning_window_sigma);
                    
                    // taking the neurons that were active within a gaussian learning window
                    if (gaussian_window >= 0.01) {
                        auto& presynaptic_neuron = network->getNeurons()[input->getPresynapticNeuronID()];
                        
                        // calculating the time difference
                        double time_difference = timestamp - spike_arrival_time;
                        
                        // saving information for the corresponding logger
                        time_differences.emplace_back(time_difference);
                        plastic_coordinates[0].emplace_back(presynaptic_neuron->getXYCoordinates().first);
                        plastic_coordinates[1].emplace_back(presynaptic_neuron->getXYCoordinates().second);
                        plastic_coordinates[2].emplace_back(presynaptic_neuron->getRfCoordinates().first);
                        plastic_coordinates[3].emplace_back(presynaptic_neuron->getRfCoordinates().second);
                        
                        // change delay according to the time difference
                        float delta_delay = 0;
                        if (time_difference > 0) {
                            delta_delay = learning_rate * (1/(input->getSynapseTimeConstant() - postsynapticNeuron->getMembraneTimeConstant())) * postsynapticNeuron->getCurrent() * (std::exp(-time_difference/input->getSynapseTimeConstant()) - std::exp(-time_difference/postsynapticNeuron->getMembraneTimeConstant()));
                            input->setDelay(delta_delay);
                        } else if (time_difference < 0) {
                            delta_delay = - learning_rate * (1/(input->getSynapseTimeConstant() - postsynapticNeuron->getMembraneTimeConstant())) * postsynapticNeuron->getCurrent() * (std::exp(time_difference/input->getSynapseTimeConstant()) - std::exp(time_difference/postsynapticNeuron->getMembraneTimeConstant()));
                            input->setDelay(delta_delay);
                        }
                        
                        // decrease the synaptic efficacy as the delays converge
                        input->setSynapticEfficacy(-std::exp(-time_difference * time_difference)+1, false);
                        
                        // increasing weights depending on activity, according to a gaussian on the time difference
                        float delta_weight = learning_rate * gaussian_distribution(time_difference, 0, weight_learning_window_sigma);
                        input->setWeight(delta_weight);
                        
                        if (network->getVerbose() >= 1) {
                            std::cout << timestamp << " " << input->getPresynapticNeuronID() << " " << input->getPostsynapticNeuronID() << " time difference: " << time_difference << " delay change: " << delta_delay << " delay: " << input->getDelay() << " synaptic efficacy: " << input->getSynapticEfficacy() << std::endl;
                        }
                    }
                    
                    // calculating weight normaliser
                    weight_normaliser += input->getWeight();
                }
            }
            
            // normalising synaptic weights
            for (auto& input: postsynapticNeuron->getDendriticTree()) {
                if (input->getWeight() > 0) {
                    input->setWeight(input->getWeight() / weight_normaliser, false);
                    if (network->getVerbose() >= 1) {
                        std::cout << input->getPresynapticNeuronID() << "->" << input->getPostsynapticNeuronID() << " weight: " << input->getWeight() << std::endl;
                    }
                }
            }
            
            // saving into the neuron's logger if the logger exists
            for (auto& addon: postsynapticNeuron->getRelevantAddons()) {
                if (MyelinPlasticityLogger* myelinLogger = dynamic_cast<MyelinPlasticityLogger*>(addon)) {
                    dynamic_cast<MyelinPlasticityLogger*>(addon)->myelinPlasticityEvent(timestamp, postsynapticNeuron, network, time_differences, plastic_coordinates);
                }
            }
        }
        
        inline float gaussian_distribution(float x, float mu, float sigma) {
            return 12.533 * sigma / 5 * std::exp(- 0.5 * std::pow((x - mu)/sigma, 2)) / (sigma * std::sqrt(2 * M_PI));
        }
        
	protected:
	
		// ----- LEARNING RULE PARAMETERS -----
        int              delay_learning_window_sigma;
        int              weight_learning_window_sigma;
        float            learning_rate;
	};
}
