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
        MyelinPlasticity(float _learning_window_sigma=1, float _learning_rate=1) :
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

            // forcing the neuron to be a LIF
            LIF* n = dynamic_cast<LIF*>(postsynapticNeuron);
            std::vector<double>   input_times;
            std::vector<Synapse*> input_synapses;
            
            // saving relevant synapses and their spike times
            for (auto& input: n->getDendriticTree()) {
                auto& presynapticNeuron = network->getNeurons()[input->getPresynapticNeuronID()];
                
                // ignoring inhibitory and dead synapses, also making sure to only take the recently active synapses
                if (input->getWeight() > 0 && presynapticNeuron->getEligibilityTrace() > 0.1) {
                    input_times.emplace_back(input->getPreviousInputTime());
                    input_synapses.emplace_back(input);
                }
            }
            
            // calculating the optimal time for fast convergence
            auto t_desired = compute_median(input_times);
            
            for (auto i=0; i<input_synapses.size(); i++) {
                auto& presynapticNeuron = network->getNeurons()[input_synapses[i]->getPresynapticNeuronID()];
                
                // compute time differences
                double time_difference = timestamp - input_times[i] - input_synapses[i]->getDelay();
                
                // change delays according to the time differences
                float delta_delay = 0;
                if (time_difference > 0) {
                    std::cout << n->getCurrent() << std::endl;
                    delta_delay = learning_rate* (1/(n->getDecayCurrent()-n->getDecayPotential())) * n->getCurrent() * (std::exp(-time_difference/n->getDecayCurrent()) - std::exp(-time_difference/n->getDecayPotential()));
                    input_synapses[i]->setDelay(delta_delay);
                    
                    if (network->getVerbose() >= 1) {
                        std::cout << timestamp << " " << presynapticNeuron->getNeuronID() << " " << n->getNeuronID() << " time difference: " << time_difference << " delay change: " << delta_delay << std::endl;
                    }
                } else if (time_difference < 0) {
                    delta_delay = - learning_rate * (1/(n->getDecayCurrent()-n->getDecayPotential())) * n->getCurrent() * (std::exp(time_difference/n->getDecayCurrent()) - std::exp(time_difference/n->getDecayPotential()));
                    input_synapses[i]->setDelay(delta_delay);

                    if (network->getVerbose() >= 1) {
                        std::cout << timestamp << " " << presynapticNeuron->getNeuronID() << " " << n->getNeuronID() << " time difference: " << time_difference << " delay change: " << delta_delay << std::endl;
                    }
                }
                
                // change learning rate depending on the convergence
//                input_synapses[i]->setSynapticEfficacy(-std::exp(-time_difference * time_difference) + 1);
                
                // modify the weights and normalise according to the gaussian
            }
            
//            std::vector<double> timeDifferences;
//            std::vector<int> plasticID;
//            std::vector<std::vector<int>> plasticCoordinates(4);
//            if (network->getVerbose() >= 1) {
//                std::cout << "New learning epoch at t=" << timestamp << std::endl;
//            }
//
//            for (auto& inputSynapse: n->getDendriticTree()) {
//                // discarding inhibitory synapses
//                if (inputSynapse->getWeight() >= 0) {
//                    auto& presynapticNeuron = network->getNeurons()[inputSynapse->getPresynapticNeuronID()];
//
//                    if (presynapticNeuron->getEligibilityTrace() > 0.1) {
//                        // saving relevant information in vectors for potential logging
//                        plasticID.push_back(presynapticNeuron->getNeuronID());
//                        plasticCoordinates[0].push_back(presynapticNeuron->getXYCoordinates().first);
//                        plasticCoordinates[1].push_back(presynapticNeuron->getXYCoordinates().second);
//                        plasticCoordinates[2].push_back(presynapticNeuron->getRfCoordinates().first);
//                        plasticCoordinates[3].push_back(presynapticNeuron->getRfCoordinates().second);
//                        timeDifferences.push_back(timestamp - inputSynapse->getPreviousInputTime() - inputSynapse->getDelay());
//
//                        float delta_delay = 0;
//
//                        if (timeDifferences.back() > 0) {
//                            delta_delay = delay_lambda*(1/(n->getDecayCurrent()-n->getDecayPotential())) * n->getCurrent() * (std::exp(-delay_alpha*timeDifferences.back()/n->getDecayCurrent()) - std::exp(-delay_alpha*timeDifferences.back()/n->getDecayPotential()));
//
//                            inputSynapse->setDelay(delta_delay);
//
//                            if (network->getVerbose() >= 1) {
//                                std::cout << timestamp << " " << presynapticNeuron->getNeuronID() << " " << n->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << delta_delay << std::endl;
//                            }
//                        } else if (timeDifferences.back() < 0) {
//                            delta_delay = -delay_lambda*((1/(n->getDecayCurrent()-n->getDecayPotential())) * n->getCurrent() * (std::exp(delay_alpha*timeDifferences.back()/n->getDecayCurrent()) - std::exp(delay_alpha*timeDifferences.back()/n->getDecayPotential())));
//
//                            inputSynapse->setDelay(delta_delay);
//
//                            if (network->getVerbose() >= 1) {
//                                std::cout << timestamp << " " << presynapticNeuron->getNeuronID() << " " << n->getNeuronID() << " time difference: " << timeDifferences.back() << " delay change: " << delta_delay << std::endl;
//                            }
//                        }
//                    }
//                }
//            }
//
//            // shifting weights to be equal to the number of plastic neurons
//            float desiredWeight = 1./plasticID.size();
//
//            for (auto i=0; i<postsynapticNeuron->getDendriticTree().size(); i++) {
//                auto& dendrite = postsynapticNeuron->getDendriticTree()[i];
//                // discarding inhibitory synapses
//                if (dendrite->getWeight() >= 0) {
//                    int ID = network->getNeurons()[dendrite->getPresynapticNeuronID()]->getNeuronID();
//                    if (std::find(plasticID.begin(), plasticID.end(), ID) != plasticID.end()) {
//                        float weightDifference = desiredWeight - dendrite->getWeight();
//                        float change = - std::exp(- (weight_alpha*weightDifference) * (weight_alpha*weightDifference)) + 1;
//                        if (weightDifference >= 0) {
//                            dendrite->setWeight(weight_lambda*change * (1 - dendrite->getWeight()));
//                        } else {
//                            dendrite->setWeight(-weight_lambda*change * (1 - dendrite->getWeight()));
//                        }
//                    } else {
//                        dendrite->setWeight(-weight_lambda * (1 - dendrite->getWeight()));
//                    }
//                }
//            }
//
//            for (auto& addon: network->getAddons()) {
//                if (MyelinPlasticityLogger* myelinLogger = dynamic_cast<MyelinPlasticityLogger*>(addon.get())) {
//                    dynamic_cast<MyelinPlasticityLogger*>(addon.get())->myelinPlasticityEvent(timestamp, postsynapticNeuron, network, timeDifferences, plasticCoordinates);
//                }
//            }
        }
		
        double compute_median(std::vector<double> input_times) {
            if (input_times.size() % 2 == 0) {
                const auto median_it1 = input_times.begin() + input_times.size() / 2 - 1;
                const auto median_it2 = input_times.begin() + input_times.size() / 2;
                
                std::nth_element(input_times.begin(), median_it1 , input_times.end());
                const auto e1 = *median_it1;
                
                std::nth_element(input_times.begin(), median_it2 , input_times.end());
                const auto e2 = *median_it2;
                
                return (e1 + e2) / 2;
                
            } else {
                const auto median_it = input_times.begin() + input_times.size() / 2;
                std::nth_element(input_times.begin(), median_it , input_times.end());
                return *median_it;
            }
        }
        
        inline float gaussian_distribution(float x, float mu, float sigma) {
            return (1 / sigma * std::sqrt(2*M_PI)) * std::exp(- ((x - mu) * (x - mu))/(2*sigma*sigma));
        }
        
	protected:
	
		// ----- LEARNING RULE PARAMETERS -----
        float            learning_window_sigma;
        float            learning_rate;
	};
}
