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

#include "neurons/parrot.hpp"
#include "neurons/decision_making.hpp"
#include "neurons/cuba_lif.hpp"

#include "synapses/exponential.hpp"
#include "synapses/square.hpp"

#include "../third_party/json.hpp"

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
	
                        if (layer[i]["neuron_type"].is_number()) {

                            switch (int neuronType = layer[i]["neuron_type"].get<int>(); neuronType) {
                                // creating parrot layer
                                case 0: {
                                    layer_helper<Parrot>(layer[i]);
                                    break;
                                // creating CUBA_LIF layer
                                } case 1: {
                                    layer_helper<CUBA_LIF>(layer[i]);
                                    break;
                                // creating Decision_Making layer
                                } case 2: {
                                    layer_helper<Decision_Making>(layer[i]);
                                    break;
                                }
                            }
                        } else {
                            throw std::logic_error("neuronType should be a number. 0 for Parrot, 1 for CUBA_LIF, w for DecisionMaking");
                        }
                    }
                } else {
                    throw std::logic_error("layers have an incorrect format");
                }
                
                // adding correct parameters to the neurons, creating the dendritic and axonal synapses and setting the parameters of the axonal synapses according to the JSON file
                if (!network->get_neurons().empty()) {
                    for (auto& n: network->get_neurons()) {
                        neuron_helper(input.back()["neurons"][n->get_neuron_id()], n.get());
                    }
                }
                
                // setting the parameters for the dendritic connections according to the JSON file+
                if (!network->get_neurons().empty()) {
                    for (auto& n: network->get_neurons()) {
                        auto& dendriticSynapse = input.back()["neurons"][n->get_neuron_id()]["dendritic_synapses"];
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

                                n->get_dendritic_tree()[i]->set_weight(weight);
                                n->get_dendritic_tree()[i]->set_delay(delay);
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
        void neuron_helper(nlohmann::json& input, Neuron* n) {
            // common neuron parameters
            if (input["rf_id"].is_number()) {
                n->set_rf_id(input["rf_id"].get<int>());
            }
            
            if (input["xy_coordinates"].is_array() && input["xy_coordinates"].size() == 2) {
                int x = input["xy_coordinates"][0].get<int>();
                int y = input["xy_coordinates"][1].get<int>();
                n->set_xy_coordinates(x, y);
            }
            
            if (input["trace_time_constant"].is_number()) {
                n->set_trace_time_constant(input["trace_time_constant"].get<float>());
            }
            
            if (input["resting_potential"].is_number()) {
                n->set_resting_potential(input["resting_potential"].get<float>());
            }
            
            if (input["threshold"].is_number()) {
                n->set_threshold(input["threshold"].get<float>());
            }
            
            if (input["refractory_period"].is_number()) {
                n->set_refractory_period(input["refractory_period"].get<int>());
            }
            
            if (input["membrane_time_constant"].is_number()) {
                n->set_membrane_time_constant(input["membrane_time_constant"].get<float>());
            }
            
            if (input["capacitance"].is_number()) {
                n->set_capacitance(input["capacitance"].get<float>());
            }
            
            if (input["leakage_conductance"].is_number()) {
                n->set_leakage_conductance(input["leakage_conductance"].get<float>());
            }
            
            if (input["class_label"].is_string()) {
                dynamic_cast<Decision_Making*>(n)->set_class_label(input["class_label"].get<std::string>());
            }

            // specific neuron parameters
            if (input["type"].is_number()) {
                switch (int type = input["type"].get<int>(); type) {
                    // CUBA_LIF neuron
                    case 1: {
                        capture_CUBA_LIF_parameters<CUBA_LIF>(input, n);
                        break;
                    } 
                }
            }

            // Connecting the network and setting the parameters for axonal synapses
            auto& axonalSynapse = input["axonal_synapses"];
            if (axonalSynapse.is_array() && !axonalSynapse.empty()) {

                for (auto i=0; i<axonalSynapse.size(); i++) {
                    float weight = 0;
                    if (axonalSynapse[i]["weight"].is_number()) {
                        weight = axonalSynapse[i]["weight"].get<float>();
                    } else {
                        throw std::logic_error("axonal synapse weight incorrectly formatted");
                    }

                    float delay = 0;
                    if (axonalSynapse[i]["delay"].is_number()) {
                        delay = axonalSynapse[i]["delay"].get<float>();
                    } else {
                        throw std::logic_error("axonal synapse weight incorrectly formatted");
                    }
                    
                    if (axonalSynapse[i]["postsynaptic_neuron"].is_number()) {
                        float synapseTimeConstant = 0;
                        
                        switch (int json_id = axonalSynapse[i]["json_id"].get<int>(); json_id) {
                            case 0: {
                            n->make_synapse<Synapse>(network->get_neurons()[axonalSynapse[i]["postsynaptic_neuron"].get<int>()].get(), 100., weight, delay);
                                
                                break;
                            } case 1: {
                                if (axonalSynapse[i]["synapse_time_constant"].is_number()) {
                                    synapseTimeConstant = axonalSynapse[i]["synapse_time_constant"].get<float>();
                                } else {
                                    throw std::logic_error("exponential synaptic time constant incorrectly formatted");
                                }
                                
                            n->make_synapse<Exponential>(network->get_neurons()[axonalSynapse[i]["postsynaptic_neuron"].get<int>()].get(), 100., weight, delay, synapseTimeConstant);
                                
                                break;
                            } case 2:
                                if (axonalSynapse[i]["synapse_time_constant"].is_number()) {
                                    synapseTimeConstant = axonalSynapse[i]["synapse_time_constant"].get<float>();
                                } else {
                                    throw std::logic_error("pulse synaptic time constant incorrectly formatted");
                                }
                                
                            n->make_synapse<Square>(network->get_neurons()[axonalSynapse[i]["postsynaptic_neuron"].get<int>()].get(), 100., weight, delay, synapseTimeConstant);
                                
                                break;
                        }
                    } else {
                        throw std::logic_error("postsynapticNeuron incorrectly formatted");
                    }
                }
            }
        }
        
        // builds a layer according to the parameter in the JSON network save file
        template<typename T>
        void layer_helper(nlohmann::json& input) {
            // vector of learning rule addons for a layer
            std::vector<Addon*> learningRules;
            
            if (input["neuron_number"].is_number() & input["sublayer_number"].is_number()) {
                // getting number of neurons
                int neuronNumber = input["neuron_number"].get<int>();
                
                // getting number of sublayers
                int sublayerNumber = input["sublayer_number"].get<int>();
                
                if (input["width"].is_number() && input["height"].is_number()) {
                    int width = input["width"].get<int>();
                    int height = input["height"].get<int>();
					
                    // checking if 1D or 2D layer
                    if (width == -1 && height == -1) {
                        network->make_layer<T>(neuronNumber, {});
                    } else {
                        network->make_grid<T>(width, height, sublayerNumber, {});
                    }
                } else {
                    throw std::logic_error("incorrect format: width and height should be numbers");
                }
                
            } else {
                throw std::logic_error("incorrect format: neuron_number and sublayer_number should be numbers");
            }
        }
        
        // parameters specific for the CUBA_LIF parent class
        template<typename T>
        void capture_CUBA_LIF_parameters(nlohmann::json& input, Neuron* n) {
            if (input["wta"].is_boolean()) {
                dynamic_cast<T*>(n)->set_wta(input["wta"].get<bool>());
            }
            
            if (input["bursting_activity"].is_boolean()) {
                dynamic_cast<T*>(n)->set_bursting_activity(input["bursting_activity"].get<bool>());
            }

            if (input["decay_homeostasis"].is_number()) {
                dynamic_cast<T*>(n)->set_decay_homeostasis(input["decay_homeostasis"].get<float>());
            }

            if (input["homeostasis"].is_boolean()) {
                dynamic_cast<T*>(n)->set_homeostasis(input["homeostasis"].get<bool>());
            }
            
            if (input["homeostasis_beta"].is_number()) {
                dynamic_cast<T*>(n)->set_homeostasis_beta(input["homeostasis_beta"].get<float>());
            }

            if (input["resting_threshold"].is_number()) {
                dynamic_cast<T*>(n)->set_resting_threshold(input["resting_threshold"].get<float>());
            }
        }
        
        // ----- IMPLEMENTATON VARIABLES -----
        Network* network;
	};
}
