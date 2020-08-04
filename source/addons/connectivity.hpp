/*
 * connectivity.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 08/01/2020
 *
 * Information: tells me which neurons were active for classification
 */

#pragma once

#include <vector>
#include <string>

#include "../third_party/numpy.hpp"

namespace hummus {
    
    class Synapse;
    class Neuron;
    class Network;
    
	class Connectivity : public Addon {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        Connectivity() = default;
        virtual ~Connectivity(){}
        
        // ----- PUBLIC METHODS -----
		void on_completed(Network* network) override {
            // find out if output neurons are used in classification
            std::vector<int> output_neurons;
            int number_of_neurons = 0;
            if (network->get_decision_making()) {
                auto& neurons = network->get_layers()[network->get_decision_parameters().layer_number-1].neurons;
                number_of_neurons = static_cast<int>(neurons.size());
                for (auto& n: neurons) {
                    output_neurons.emplace_back(static_cast<int>(n));
                    
                    if (network->get_neurons()[n]->get_axon_terminals().empty()) {
                        output_neurons.emplace_back(0);
                    } else {
                        output_neurons.emplace_back(1);
                    }
                }
            } else if (network->get_logistic_regression()) {
                auto& neurons = network->get_layers()[network->get_decision_parameters().layer_number-2].neurons;
                number_of_neurons = static_cast<int>(neurons.size());
                for (auto& n: neurons) {
                    output_neurons.emplace_back(static_cast<int>(n));
                    if (network->get_neurons()[n]->get_axon_terminals().empty()) {
                        output_neurons.emplace_back(0);
                    } else {
                        output_neurons.emplace_back(1);
                    }
                }
            }
            
            // saving the vector to disk
            if (network->get_decision_making() || network->get_logistic_regression()) {
                const int output_shape[2] = {number_of_neurons,2};
                aoba::SaveArrayAsNumpy("active_neurons.npy", false, 2, &output_shape[0], &output_neurons[0]);
            }
		}
	};
}
