/*
 * parrot.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/01/2019
 *
 * Information: Parrot neurons take in spikes or events and instantly propagate them in the network. The potential does not decay.
 *
 * NEURON TYPE 0 (in JSON SAVE FILE)
 */

#pragma once

#include "../../third_party/json.hpp"

namespace hummus {
    
    class Synapse;
    class Neuron;
    class Network;
    
	class Parrot : public Neuron {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        Parrot(int _neuronID, int _layerID, int _sublayerID, int _rf_id,  std::pair<int, int> _xyCoordinates, int _refractoryPeriod=0, float _traceTimeConstant=20, float _threshold=-50, float _restingPotential=-70) :
                Neuron(_neuronID, _layerID, _sublayerID, _rf_id, _xyCoordinates, _refractoryPeriod, 200, 10, _traceTimeConstant, _threshold, _restingPotential, "") {
            inv_trace_tau = 1. / _traceTimeConstant;
        }
		
		virtual ~Parrot(){}
		
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
            
            if (network->is_asynchronous()) {
                timestep = timestamp - previous_spike_time;
            }
            
            // checking if the neuron is in a refractory period
            if (timestep >= refractory_period) {
                active = true;
            }
            
            // trace decay
            trace -= timestep * inv_trace_tau;
            if (trace < 0) {
                trace = 0;
            }
            
            // instantly making the input neuron fire at every input spike
            if (s && active) {
                potential = threshold;
                trace = 1;
                
                if (network->get_verbose() == 2) {
                    std::cout << "t=" << timestamp << " " << neuron_id << " w=" << s->get_weight() << " d=" << s->get_delay() << " --> INPUT" << std::endl;
                }
                
                for (auto& addon: relevant_addons) {
                    addon->neuron_fired(timestamp, s, this, network);
                }
                
                if (network->get_main_thread_addon()) {
                    network->get_main_thread_addon()->neuron_fired(timestamp, s, this, network);
                }
                
                for (auto& axonTerminal : axon_terminals) {
                    auto& post_synaptic_layer = network->get_layers()[network->get_neurons()[axonTerminal->get_postsynaptic_neuron_id()]->get_layer_id()];
                    if (post_synaptic_layer.active) {
                        network->inject_spike(spike{timestamp + axonTerminal->get_delay(), axonTerminal.get(), spike_type::generated});
                    }
                }
                
                request_learning(timestamp, s, this, network);
                
                if (network->get_main_thread_addon()) {
                    network->get_main_thread_addon()->status_update(timestamp, this, network);
                }
                
                for (auto& addon: relevant_addons) {
                    addon->status_update(timestamp, this, network);
                }
                
                previous_spike_time = timestamp;
                potential = resting_potential;
                active = false;
                
            } else {
                // if the network is not asynchronous and we're on a different timestamp (handling spikes that fire at the same time)
                if (!network->is_asynchronous() && timestep > 0) {
                    if (network->get_main_thread_addon()) {
                        network->get_main_thread_addon()->status_update(timestamp, this, network);
                    }
                    
                    for (auto& addon: relevant_addons) {
                        addon->status_update(timestamp, this, network);
                    }
                }
            }
		}
        
        // write neuron parameters in a JSON format
        virtual void to_json(nlohmann::json& output) override{
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
        
        // ----- PARROT PARAMETERS -----
        float   inv_trace_tau;
	};
}
