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

#include "neurons/input.hpp"
#include "neurons/decisionMaking.hpp"
#include "neurons/LIF.hpp"

#include "learningRules/stdp.hpp"
#include "learningRules/timeInvariantSTDP.hpp"
#include "learningRules/myelinPlasticity.hpp"
#include "learningRules/rewardModulatedSTDP.hpp"

#include "synapses/exponential.hpp"
#include "synapses/dirac.hpp"
#include "synapses/pulse.hpp"

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
                                layerHelper<Input>(layer[i]);
                            // creating LIF layer
                            } else if (neuronType == 1) {
                                layerHelper<LIF>(layer[i]);
                            // creating DecisionMaking layer
                            } else if (neuronType == 2) {
                                layerHelper<DecisionMaking>(layer[i]);
                            }
                        } else {
                            throw std::logic_error("neuronType should be a number. 0 for InputNeuron, 1 for LIF, 2 for IF, 3 for DecisionMakingNeuron");
                        }
                    }
                } else {
                    throw std::logic_error("layers have an incorrect format");
                }
                
                // adding correct parameters to the neurons, creating the dendritic and axonal synapses and setting the parameters of the axonal synapses according to the JSON file
                if (!network->getNeurons().empty()) {
                    for (auto& n: network->getNeurons()) {
                        neuronHelper(input.back()["neurons"][n->getNeuronID()], n.get());
                    }
                }
                
                // setting the parameters for the dendritic connections according to the JSON file+
                if (!network->getNeurons().empty()) {
                    for (auto& n: network->getNeurons()) {
                        auto& dendriticSynapse = input.back()["neurons"][n->getNeuronID()]["dendriticSynapses"];
                        if (dendriticSynapse.is_array() && !dendriticSynapse.empty()) {
                            for (auto i=0; i<dendriticSynapse.size(); i++) {
                                float weight = 0;
                                if (dendriticSynapse[i]["weight"].is_number()) {
                                    weight = dendriticSynapse[i]["weight"].get<float>();
                                } else {
                                    throw std::logic_error("dendritic synapse weight incorrectly formatted");
                                }

                                float delay = 0;
                                if (dendriticSynapse[i]["delay"].is_number()) {
                                    delay = dendriticSynapse[i]["delay"].get<float>();
                                } else {
                                    throw std::logic_error("dendritic synapse weight incorrectly formatted");
                                }

                                n->getDendriticTree()[i]->setWeight(weight, false);
                                n->getDendriticTree()[i]->setDelay(delay, false);
                            }
                        }
                    }
                }
                
            } else {
                throw std::logic_error("incorrect format");
            }
        }
        
    protected:
        
        // changes the default parameters of a neuron to correspond to the ones in the JSON network save file
        void neuronHelper(nlohmann::json& input, Neuron* n) {
            // common neuron parameters
            if (input["receptiveFieldCoordinates"].is_array() && input["receptiveFieldCoordinates"].size() == 2) {
                int row = input["receptiveFieldCoordinates"][0].get<int>();
                int col = input["receptiveFieldCoordinates"][1].get<int>();
                n->setRfCoordinates(row, col);
            }
            
            if (input["XYCoordinates"].is_array() && input["XYCoordinates"].size() == 2) {
                int x = input["XYCoordinates"][0].get<int>();
                int y = input["XYCoordinates"][1].get<int>();
                n->setXYCoordinates(x, y);
            }
            
            if (input["eligibilityDecay"].is_number()) {
                n->setEligibilityDecay(input["eligibilityDecay"].get<float>());
            }
            
            if (input["restingPotential"].is_number()) {
                n->setRestingPotential(input["restingPotential"].get<float>());
            }
            
            if (input["threshold"].is_number()) {
                n->setThreshold(input["threshold"].get<float>());
            }
            
            // specific neuron parameters
            if (input["Type"].is_number()) {
                int type = input["Type"].get<int>();
                // input neuron
                if (type == 0) {
                    if (input["refractoryPeriod"].is_number()) {
                        dynamic_cast<Input*>(n)->setRefractoryPeriod(input["refractoryPeriod"].get<float>());
                    }

                // LIF neuron
                } else if (type == 1) {
                    captureLIFParameters<LIF>(input, n);
                // DecisionMaking neuron
                } else if (type == 2) {

                    captureLIFParameters<DecisionMaking>(input, n);

                    if (input["classLabel"].is_string()) {
                        dynamic_cast<DecisionMaking*>(n)->setClassLabel(input["refractoryPeriod"].get<std::string>());
                    }

                }
            }

            // Connecting the network and setting the parameters for axonal synapses
            auto& axonalSynapse = input["axonalSynapses"];
            if (axonalSynapse.is_array() && !axonalSynapse.empty()) {

                for (auto i=0; i<axonalSynapse.size(); i++) {
                    double weight = 0;
                    if (axonalSynapse[i]["weight"].is_number()) {
                        weight = axonalSynapse[i]["weight"].get<float>();
                    } else {
                        throw std::logic_error("axonal synapse weight incorrectly formatted");
                    }

                    double delay = 0;
                    if (axonalSynapse[i]["delay"].is_number()) {
                        delay = axonalSynapse[i]["delay"].get<float>();
                    } else {
                        throw std::logic_error("axonal synapse weight incorrectly formatted");
                    }

                    if (axonalSynapse[i]["postNeuronID"].is_number()) {
                        if (axonalSynapse[i]["type"].get<int>() == 0) {
                            n->makeSynapse<Dirac>(network->getNeurons()[axonalSynapse[i]["postNeuronID"].get<int>()].get(), 100, weight, delay);
                        } else if (axonalSynapse[i]["type"].get<int>() == 1){
                            n->makeSynapse<Exponential>(network->getNeurons()[axonalSynapse[i]["postNeuronID"].get<int>()].get(), 100, weight, delay);
                        } else if (axonalSynapse[i]["type"].get<int>() == 2) {
                            n->makeSynapse<Pulse>(network->getNeurons()[axonalSynapse[i]["postNeuronID"].get<int>()].get(), 100, weight, delay);
                        }
                    } else {
                        throw std::logic_error("postNeuronID incorrectly formatted");
                    }
                }
            }
        }
        
        // builds a layer according to the parameter in the JSON network save file
        template<typename T>
        void layerHelper(nlohmann::json& input) {
            // vector of learning rule addons for a layer
            std::vector<Addon*> learningRules;
            
            if (input["neuronNumber"].is_number() & input["sublayerNumber"].is_number()) {
                // getting number of neurons
                int neuronNumber = input["neuronNumber"].get<int>();
                
                // getting number of sublayers
                int sublayerNumber = input["sublayerNumber"].get<int>();
                
                if (input["width"].is_number() && input["height"].is_number()) {
                    int width = input["width"].get<int>();
                    int height = input["height"].get<int>();
					
                    // checking if 1D or 2D layer
                    if (width == -1 && height == -1) {
                        network->makeLayer<T>(neuronNumber, {});
                    } else {
                        network->makeGrid<T>(width, height, sublayerNumber, {});
                    }
                } else {
                    throw std::logic_error("incorrect format: width and height should be numbers");
                }
                
            } else {
                throw std::logic_error("incorrect format: neuronNumber and sublayerNumber should be numbers");
            }
        }
        
        // parameters specific for the LIF parent class
        template<typename T>
        void captureLIFParameters(nlohmann::json& input, Neuron* n) {
            if (input["refractoryPeriod"].is_number()) {
                dynamic_cast<T*>(n)->setRefractoryPeriod(input["refractoryPeriod"].get<float>());
            }
            
            if (input["burstingActivity"].is_boolean()) {
                dynamic_cast<T*>(n)->setBurstingActivity(input["burstingActivity"].get<bool>());
            }

            if (input["decayHomeostasis"].is_number()) {
                dynamic_cast<T*>(n)->setDecayHomeostasis(input["decayHomeostasis"].get<float>());
            }

            if (input["decayPotential"].is_number()) {
                dynamic_cast<T*>(n)->setDecayPotential(input["decayPotential"].get<float>());
            }
            
            if (input["decayCurrent"].is_number()) {
                dynamic_cast<T*>(n)->setDecayCurrent(input["decayCurrent"].get<float>());
            }

            if (input["homeostasis"].is_boolean()) {
                dynamic_cast<T*>(n)->setHomeostasis(input["homeostasis"].get<bool>());
            }
            
            if (input["homeostasisBeta"].is_number()) {
                dynamic_cast<T*>(n)->setHomeostasisBeta(input["homeostasisBeta"].get<float>());
            }

            if (input["restingThreshold"].is_number()) {
                dynamic_cast<T*>(n)->setRestingThreshold(input["restingThreshold"].get<float>());
            }
            
            if (input["wta"].is_boolean()) {
                dynamic_cast<T*>(n)->setWTA(input["wta"].get<bool>());
            }
        }
        
        // ----- IMPLEMENTATON VARIABLES -----
        Network* network;
	};
}
