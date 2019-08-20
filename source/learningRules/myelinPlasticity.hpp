/*
 * myelinPlasticity.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: The MyelinPlasticity learning rule compatible only with leaky integrate-and-fire neurons.
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
        MyelinPlasticity(int _time_constant=10, int _learning_window=20, float _learning_rate=1, float _alpha_plus=0.2, float _alpha_minus=-0.08, float _beta_plus=1, float _beta_minus=0) :
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
            std::vector<Synapse*> accepted_synapses;
            
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
                    if (inputNeuron->getTrace() > 0 && gaussian_window >= 0.01) {
                        accepted_synapses.emplace_back(input);
                        
                        // increasing the threshold if the trace is too high
                        if (inputNeuron->getTrace() >= 1) {
                            auto updated_threshold = inputNeuron->getThreshold() + 2;
                            inputNeuron->setThreshold(updated_threshold);
                        }
                        
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

                        weight_normaliser += input->getWeight();
                        
                        if (network->getVerbose() >= 1) {
                            std::cout << " inside learning window " << spike_arrival_time << " " << input->getPresynapticNeuronID() << " " << input->getPostsynapticNeuronID() << " time difference: " << time_difference << " delay change: " << delta_delay << " delay: " << input->getDelay() << " weight change: " << delta_weight << " weight " << input->getWeight() << " trace " << inputNeuron->getTrace() << " threshold " << inputNeuron->getThreshold() << std::endl;
                        }
                        
                    // taking the input neurons that are outside the learning window
                    } else if (inputNeuron->getTrace() > 0 && gaussian_window < 0.01){
                        // decreasing the threshold if the trace is too high
                        if (inputNeuron->getThreshold() > -55) {
                            auto updated_threshold = inputNeuron->getThreshold() - 2;
                            inputNeuron->setThreshold(updated_threshold);
                        }

                        // long-term potentiation on weights
                        float delta_weight = (alpha_plus * std::exp(- beta_plus * input->getWeight())) * input->getWeight() * (1 - input->getWeight());
                        input->setWeight(delta_weight);

                        weight_normaliser += input->getWeight();

                        if (network->getVerbose() >= 1) {
                            std::cout << " outside learning window " << spike_arrival_time << " " << input->getPresynapticNeuronID() << " " << input->getPostsynapticNeuronID() << " weight change: " << delta_weight << " weight " << input->getWeight() << " trace " << inputNeuron->getTrace() << " threshold " << inputNeuron->getThreshold() << std::endl;
                        }
                    // taking the input neurons that didn't fire
                    } else {
                        // long-term depression on weights
                        float delta_weight = (alpha_minus * std::exp(- beta_minus * (1 - input->getWeight()))) * input->getWeight() * (1 - input->getWeight());
                        input->setWeight(delta_weight);

                        if (network->getVerbose() >= 1) {
                            std::cout << " never fired " << spike_arrival_time << " " << input->getPresynapticNeuronID() << " " << input->getPostsynapticNeuronID() << " weight change: " << delta_weight << " weight " << input->getWeight() << " trace " << inputNeuron->getTrace() << " threshold " << inputNeuron->getThreshold() << std::endl;
                        }
                    }
                    
                    // resetting trace for the input neuron
                    inputNeuron->setTrace(0);
                }
            }
            
            for (auto& input: postsynapticNeuron->getDendriticTree()) {
                if (input->getType() == synapseType::excitatory && weight_normaliser > 0) {
                    // normalising synaptic weights only when pattern has more than 1 neuron responding
                    input->setWeight(input->getWeight()/ weight_normaliser, false);

                    // printing the weights
                    if (network->getVerbose() >= 1) {
                        std::cout << input->getPresynapticNeuronID() << "->" << input->getPostsynapticNeuronID() << " weight: " << input->getWeight() << std::endl;
                    }
                }
            }
        
            // saving into the neuron's logger if the logger exists
            for (auto& addon: postsynapticNeuron->getRelevantAddons()) {
                if (MyelinPlasticityLogger* myelinLogger = dynamic_cast<MyelinPlasticityLogger*>(addon)) {
                    dynamic_cast<MyelinPlasticityLogger*>(addon)->myelinPlasticityEvent(timestamp, postsynapticNeuron, network, time_differences, accepted_synapses);
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
