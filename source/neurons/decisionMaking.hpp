/*
 * decisionMaking.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/01/2019
 *
 * Information: Decision-making neurons act as our classifier, roughly approximating a histogram activity-dependent classification. They should always be on the last layer of a network.
 * 
 * NEURON TYPE 2 (in JSON SAVE FILE)
 */

#pragma once

#include "../../third_party/json.hpp"

namespace hummus {
    
    class Synapse;
    class Neuron;
    class Network;
    
	class DecisionMaking : public Neuron {
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        DecisionMaking(int _neuronID, int _layerID, int _sublayerID, std::pair<int, int> _rfCoordinates,  std::pair<float, float> _xyCoordinates, std::string _classLabel="", int _refractoryPeriod=10, float _conductance=200, float _leakageConductance=10, float _traceTimeConstant=20, float _threshold=-50, float _restingPotential=-70) :
                Neuron(_neuronID, _layerID, _sublayerID, _rfCoordinates, _xyCoordinates, _refractoryPeriod, _conductance, _leakageConductance, _traceTimeConstant, _threshold, _restingPotential, _classLabel),
                active(true),
                inhibition_time(0) {
                    
            // DecisionMaking neuron type = 2 for JSON save
                    neuron_type = 2;
        }
		
		virtual ~DecisionMaking(){}
		
        // ----- PUBLIC DECISION MAKING NEURON METHODS -----
        virtual void initialisation(Network* network) override {
            // searching for addons that are relevant to this neuron. if addons do not have a mask they are automatically relevant / not filtered out
            for (auto& addon: network->get_addons()) {
                if (addon->get_mask().empty() && !addon->no_automatic_include()) {
                    add_relevant_addon(addon.get());
                } else {
                    auto it = std::find(addon->get_mask().begin(), addon->get_mask().end(), static_cast<size_t>(neuron_id));
                    if (it != addon->get_mask().end()) {
                        add_relevant_addon(addon.get());
                    }
                }
            }
        }
        
        virtual void update(double timestamp, Synapse* s, Network* network, spike_type type) override {
            // checking if the neuron is in a refractory period
            if (timestamp - inhibition_time >= refractory_period) {
                active = true;
            }
            
            if (type == spike_type::decision) {
                if (active && intensity > 0) {
                    // function that converts the intensity to a delay
                    float intensity_to_latency = 10 * 1 - std::exp(- intensity/dendritic_tree.size());
                    
                    // make the neuron fire so we can get the decision
                    potential = threshold;
                    
                    if (network->get_verbose() == 1) {
                        std::cout << "t=" << timestamp << " class " << class_label << " --> DECISION" << std::endl;
                    }
                    
                    for (auto& addon: relevant_addons) {
                        addon->neuron_fired(timestamp, s, this, network);
                    }
                    
                    if (network->get_main_thread_addon()) {
                        network->get_main_thread_addon()->neuron_fired(timestamp, s, this, network);
                    }
                    
                    // propagating the decision spike
                    if (!network->get_layers()[layer_id].do_not_propagate) {
                        for (auto& axonTerminal: axon_terminals) {
                            network->inject_spike(spike{timestamp + intensity_to_latency, axonTerminal.get(), spike_type::generated});
                        }
                    }
                    
                    // inhibiting the other decision_making neurons
                    winner_takes_all(timestamp, network);
                }
                
                intensity = 0;
                
            } else if (type != spike_type::none){
                ++intensity;
            }
        }
        
        // write neuron parameters in a JSON format
        virtual void to_json(nlohmann::json& output) override{
            // general neuron parameters
            output.push_back({
                {"type",neuron_type},
                {"layer_id",layer_id},
                {"sublayer_id", sublayer_id},
                {"rf_coordinates", rf_coordinates},
                {"xy_coordinates", xy_coordinates},
                {"trace_time_constant", trace_time_constant},
                {"threshold", threshold},
                {"resting_potential", resting_potential},
                {"refractory_period", refractory_period},
                {"dendritic_synapses", nlohmann::json::array()},
                {"axonal_synapses", nlohmann::json::array()},
            });
            
            // dendritic synapses (preSynapse)
            auto& dendriticSynapses = output.back()["dendritic_synapses"];
            for (auto& dendrite: dendritic_tree) {
                dendrite->to_json(dendriticSynapses);
            }
            
            // axonal synapses (postSynapse)
            auto& axonalSynapses = output.back()["axonal_synapses"];
            for (auto& axonTerminal: axon_terminals) {
                axonTerminal->to_json(axonalSynapses);
            }
        }
        
    protected:
        
        void winner_takes_all(double timestamp, Network* network) {
            for (auto& n: network->get_layers()[layer_id].neurons) {
                auto& neuron = network->get_neurons()[n];
                
                // inhibit all the other neurons in the same layer
                if (neuron->get_neuron_id() != neuron_id) {
                    dynamic_cast<DecisionMaking*>(neuron.get())->set_activity(false);
                    dynamic_cast<DecisionMaking*>(neuron.get())->set_inhibition_time(timestamp);
                }
            }
        }
        
        // ----- SETTERS AND GETTERS -----
        void set_activity(bool new_state) {
            active = new_state;
        }
        
        void set_inhibition_time(double new_time) {
            inhibition_time = new_time;
        }
        
		// ----- DECISION-MAKING NEURON PARAMETERS -----
        float    intensity;
        bool     active;
        double   inhibition_time;
	};
}
