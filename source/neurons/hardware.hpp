/*
 * hardware.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/01/2019
 *
 * Information: Parrot neurons take in spikes or events and instantly propagate them in the network. The potential does not decay.
 *
 * NEURON TYPE 3 (in JSON SAVE FILE)
 */

#pragma once

#include "../../third_party/json.hpp"

namespace hummus {
    
    class Synapse;
    class Neuron;
    class Network;
    
	class Hardware : public Neuron {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        Hardware(int _neuronID, int _layerID, int _sublayerID, int _rf_id,  std::pair<int, int> _xyCoordinates) :
                Neuron(_neuronID, _layerID, _sublayerID, _rf_id, _xyCoordinates, 0, 0, 0, 0, 0, 0, "") {
                    
            // neuron type = 3 for JSON save
            neuron_type = 3;
        }
		
		virtual ~Hardware(){}
		
		// ----- PUBLIC INPUT NEURON METHODS -----
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
        
        virtual void update(double timestamp, Synapse* s, Network* network, float timestep, spike_type type) override {
            // 1. update the synapses
            // 2. if not none and post is active send the spike to the synapses and get an estimate of the current (taking into consideration i_cancel and the scaling factor K)
            // 3. calculate potential according to the equation
            // 4. if threshold = 0 then keep increasing forever without generating spikes
            // 5. if spikes generated then send 3 spikes to terminals for the double square waveform
		}
        
        // write neuron parameters in a JSON format
        virtual void to_json(nlohmann::json& output) override {
            // general neuron parameters
            output.push_back({
                {"type",neuron_type},
                {"layer_id",layer_id},
                {"sublayer_id", sublayer_id},
                {"rf_id", rf_id},
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
        
        // loops through any learning rules and activates them
        virtual void request_learning(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            if (network->get_learning_status()) {
                for (auto& addon: relevant_addons) {
                    addon->learn(timestamp, s, postsynapticNeuron, network);
                }
            }
        }
	};
}
