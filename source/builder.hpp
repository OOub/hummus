/*
 * builder.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 05/02/2019
 *
 * Information: the Builder class takes a network instance and allows us to use the import method to build the network from a saved network JSON file
 */

#pragma once

#include "core.hpp"
#include "rand.hpp"

#include "neurons/inputNeuron.hpp"
#include "neurons/decisionMakingNeuron.hpp"
#include "neurons/LIF.hpp"
#include "neurons/IF.hpp"

#include "learningRules/stdp.hpp"
#include "learningRules/timeInvariantSTDP.hpp"
#include "learningRules/myelinPlasticity.hpp"
#include "learningRules/rewardModulatedSTDP.hpp"

#include "dependencies/json.hpp"

namespace hummus {
    
    class Network;
    
	class Builder {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        Builder(Network* _network) :
                network(_network) {}
        
		virtual ~Builder(){}
		
		// ----- PUBLIC METHODS -----
        // importing a previously created network from a JSON file
        void import(std::string filename) {
            
            nlohmann::json input;
            std::ifstream input_file(filename);
            
            if (!input_file.good()) {
                throw std::runtime_error("the file could not be opened");
            }
            
            input_file >> input;
            
            if (input.is_array()) {
                // build the layers
                if (input.back()["layers"].is_array()){
                    auto& layer = input.back()["layers"];
                    for (auto i=0; i<layer.size(); i++) {
                        if (layer[i]["neuronType"].is_number()) {
                            int neuronType = layer[i]["neuronType"].get<int>();
                            // creating input neuron layer
                            if (neuronType == 0) {
                                layerHelper<InputNeuron>(layer[i]);
                            // creating LIF layer
                            } else if (neuronType == 1) {
                                layerHelper<LIF>(layer[i]);
                            // creating IF layer
                            } else if (neuronType == 2) {
                                layerHelper<IF>(layer[i]);
                            // creating DecisionMaking layer
                            } else if (neuronType == 3) {
                                layerHelper<DecisionMakingNeuron>(layer[i]);
                            }
                        }
                    }
                }
                
                // adding correct parameters to the neurons and connecting them
                
            }
        }
        
    protected:
        
        template<typename T>
        void layerHelper(nlohmann::json& input) {
            if (input["neuronNumber"].is_number() & input["sublayerNumber"].is_number()) {
                // getting number of neurons
                int neuronNumber = input["neuronNumber"].get<int>();
                
                // getting number of sublayers
                int sublayerNumber = input["sublayerNumber"].get<int>();
                
                // getting the learning rules
                if (input["learningRules"].is_array() && !input["learningRules"].empty()) {
                    auto& rule = input["learningRules"];
                    
                    for (auto i=0; i<rule.size(); i++) {
                        
                    }
                }
                
                if (input["width"].is_number() && input["height"].is_number()) {
                    int width = input["width"].get<int>();
                    int height = input["height"].get<int>();
                    
                    // checking if 1D or 2D layer
                    if (width == -1 && height == -1) {
                        network->addLayer<T>(neuronNumber,{});
                    } else {
                        network->add2dLayer<T>(width, height, sublayerNumber, {});
                    }
                }
            }
            
        }
        
        // ----- IMPLEMENTATON VARIABLES -----
        Network* network;
	};
}
