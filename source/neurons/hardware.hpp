/*
 * hardware.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/01/2019
 *
 * Information: neuron modelled according to the ULPEC analog neuron made by the IMS at Université de Bordeaux.
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
        Hardware(int _neuronID, int _layerID, int _sublayerID, int _rf_id,  std::pair<int, int> _xyCoordinates, int _refractoryPeriod=10, float _threshold=1.2, float _restingPotential=1.1, float _i_discharge=0, float _i_cancel=0, float _scaling_factor=725) :
                Neuron(_neuronID, _layerID, _sublayerID, _rf_id, _xyCoordinates, _refractoryPeriod, 0, 0, 0, _threshold, _restingPotential, ""),
                i_cancel(_i_cancel),
                i_discharge(_i_discharge),
                scaling_factor(_scaling_factor) {
                    
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
                    if (auto it = std::find(addon->get_mask().begin(), addon->get_mask().end(), static_cast<size_t>(neuron_id)); it != addon->get_mask().end()) {
                        add_relevant_addon(addon.get());
                    }
                }
            }
        }
        
        virtual void update(double timestamp, Synapse* s, Network* network, float timestep, spike_type type) override {
            if (type != spike_type::none && active) {
                
                // handles case where the neuron never fires - for validation with the cadence experiments
                if (threshold == 0) {
                    // 1. get an estimate of the current (taking into consideration i_cancel and the scaling factor K)
                    float total_current = 0;
                    for (auto& memristor: dendritic_tree) {
                        // taking only the active memristors
                        total_current += memristor->get_synaptic_current();
                    }
                    
                    // 2. calculate potential according to the equation
                    
                } else {
                    // 1. get an estimate of the current (taking into consideration i_cancel and the scaling factor K)
                    
                    // 2. calculate potential according to the equation
                    
                    // 3. propagate spike through the axon terminals (back to the presynaptic neurons for ULPEC)
                                        
                    // 5. winner-takes-all to reset potential
                    
                    // 6. refractory period = 10 events from other output neurons
                }
            }
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
        
        float i_cancel;
        float i_discharge;
        float scaling_factor;
	};
}
