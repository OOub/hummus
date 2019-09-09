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
        Parrot(int _neuronID, int _layerID, int _sublayerID, std::pair<int, int> _rfCoordinates,  std::pair<float, float> _xyCoordinates, int _refractoryPeriod=0, float _conductance=200, float _leakageConductance=10, float _traceTimeConstant=20, float _threshold=-50, float _restingPotential=-70, std::string _classLabel="") :
                Neuron(_neuronID, _layerID, _sublayerID, _rfCoordinates, _xyCoordinates, _refractoryPeriod, _conductance, _leakageConductance, _traceTimeConstant, _threshold, _restingPotential, _classLabel),
                active(true) {}
		
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
        
        virtual void update(double timestamp, Synapse* s, Network* network, spike_type type) override {
            
            // checking if the neuron is in a refractory period
            if (timestamp - previous_spike_time >= refractory_period) {
                active = true;
            }
            
            // trace decay
            trace *= std::exp(-(timestamp - previous_spike_time)/trace_time_constant);
            
            // instantly making the input neuron fire at every input spike
            if (active) {
                potential = threshold;
                trace += 1;
                
                if (network->get_verbose() == 2) {
                    std::cout << "t=" << timestamp << " " << neuron_id << " w=" << s->get_weight() << " d=" << s->get_delay() << " --> INPUT" << std::endl;
                }
                
                for (auto& addon: relevant_addons) {
                    addon->neuron_fired(timestamp, s, this, network);
                }
                
                if (network->get_main_thread_addon()) {
                    network->get_main_thread_addon()->neuron_fired(timestamp, s, this, network);
                }
                
                if (!network->get_layers()[layer_id].do_not_propagate) {
                    for (auto& axonTerminal : axon_terminals) {
                        network->inject_spike(spike{timestamp + axonTerminal->get_delay(), axonTerminal.get(), spike_type::generated});
                    }
                }
                
                request_learning(timestamp, s, this, network);
                previous_spike_time = timestamp;
                potential = resting_potential;
                active = false;
                
                if (network->get_main_thread_addon()) {
                    network->get_main_thread_addon()->status_update(timestamp, s, this, network);
                }
            }
		}
        
        virtual void update_sync(double timestamp, Synapse* s, Network* network, double timestep, spike_type type) override {
            
            if (timestamp != 0 && timestamp - previous_spike_time == 0) {
                timestep = 0;
            }
            
            // checking if the neuron is in a refractory period
            if (timestamp - previous_spike_time >= refractory_period) {
                active = true;
            }
            
            // trace decay
            trace *= std::exp(-timestep/trace_time_constant);
            
            if (s && active) {
                potential = threshold;
                trace += 1;
                if (network->get_verbose() == 2) {
                    std::cout << "t=" << timestamp << " " << neuron_id << " w=" << s->get_weight() << " d=" << s->get_delay() << " --> INPUT" << std::endl;
                }
                
                for (auto& addon: relevant_addons) {
                    addon->neuron_fired(timestamp, s, this, network);
                }
                
                if (network->get_main_thread_addon()) {
                    network->get_main_thread_addon()->neuron_fired(timestamp, s, this, network);
                }
                
                if (!network->get_layers()[layer_id].do_not_propagate) {
                    for (auto& axonTerminal : axon_terminals) {
                        network->inject_spike(spike{timestamp + axonTerminal->get_delay(), axonTerminal.get(), spike_type::generated});
                    }
                }
                
                request_learning(timestamp, s, this, network);
                previous_spike_time = timestamp;
                potential = resting_potential;
                active = false;
            } else {
                if (timestep > 0) {
                    for (auto& addon: relevant_addons) {
                        addon->timestep(timestamp, this, network);
                    }
                    if (network->get_main_thread_addon()) {
                        network->get_main_thread_addon()->timestep(timestamp, this, network);
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
        
        // loops through any learning rules and activates them
        virtual void request_learning(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            if (network->get_learning_status()) {
                for (auto& addon: relevant_addons) {
                    addon->learn(timestamp, s, postsynapticNeuron, network);
                }
            }
        }
        
        // ----- INPUT NEURON PARAMETERS -----
        bool  active;
	};
}
