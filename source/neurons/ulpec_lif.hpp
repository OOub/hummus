/*
 * ulpec_lif.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/01/2019
 *
 * Information: neuron modeled according to the ULPEC analog neuron made by the IMS at Université de Bordeaux.
 *
 * NEURON TYPE 3 (in JSON SAVE FILE)
 */

#pragma once

#include "../../third_party/json.hpp"

namespace hummus {
    
    class Synapse;
    class Neuron;
    class Network;
    
	class ULPEC_LIF : public Neuron {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        ULPEC_LIF(int _neuronID, int _layerID, int _sublayerID, int _rf_id,  std::pair<int, int> _xyCoordinates, int _refractoryPeriod=10, double _capacitance=5e-12, double _threshold=1.2, double _restingPotential=0, double _i_discharge=12e-9, double _epsilon=0, double _scaling_factor=725, bool _potentiation_flag=true, double _tau_up=0.5, double _tau_down_event=10, double _tau_down_spike=1.5) :
                Neuron(_neuronID, _layerID, _sublayerID, _rf_id, _xyCoordinates, _refractoryPeriod, _capacitance, 0, 0, _threshold, _restingPotential, ""),
                epsilon(_epsilon),
                i_cancel(0),
                i_discharge(_i_discharge),
                scaling_factor(_scaling_factor),
                potentiation_flag(_potentiation_flag),
                tau_down_event(_tau_down_event),
                tau_down_spike(_tau_down_spike) {
                    
            // neuron type = 3 for JSON save
            neuron_type = 3;
            membrane_time_constant = _tau_up;
        }
		
		virtual ~ULPEC_LIF(){}
		
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
        
        virtual void update(double timestamp, Synapse* s, Network* network, double timestep, spike_type type) override {
            
            if (active) {
                
                // converting microseconds to seconds
                double delta_t = (timestamp - previous_input_time) * 1e-6; // converting microseconds to seconds
                    
                // checking if the neuron is in a refractory period
                if (refractory_counter >= refractory_period) { // @@DEBUG: this should be based on a counter of output neurons not on time
                    active = true;
                    refractory_counter = 0;
                }
                
                // calculating i_cancel and R_network
                double r_network = 0;
                for (auto& memristor: dendritic_tree) {
                    // taking only the inactive memristors
                    if (memristor->get_synaptic_current() <= 0) {
                        r_network = 1. / memristor->get_weight();
                    }
                }
                if (r_network > 0) {
                    i_cancel = epsilon / r_network;
                }
                
                // getting the current i_x being injected into the neuron taking into consideration i_cancel
                double i_x = 0;
                for (auto& memristor: dendritic_tree) {
                    // taking only the active memristors
                    if (memristor->get_synaptic_current() > 0) {
                        i_x += memristor->get_synaptic_current();
                    }
                }
                  
                // calculating current (i_z) taking into consideration the scaling factor
                if (i_x > i_cancel) {
                    current = (i_x - i_cancel) * 1./scaling_factor;
                } else {
                    current = 0;
                }
                
                // calculate potential according to the equation
                potential += (current * delta_t / capacitance) - (i_discharge * delta_t / capacitance);
                
                if (potential < 0) {
                    potential = 0;
                }
                
                std::cout << "t " << timestamp << " " << s->get_presynaptic_neuron_id() << "->" << neuron_id << " i_z " << current << " v_mem " << potential << std::endl;
                if (network->get_verbose() == 2) {
                    std::cout << "t " << timestamp << " " << s->get_presynaptic_neuron_id() << "->" << neuron_id << " i_z " << current << " v_mem " << potential << std::endl;
                }
                
                for (auto& addon: relevant_addons) {
                    addon->incoming_spike(timestamp, s, this, network);
                }

                if (network->get_main_thread_addon()) {
                    network->get_main_thread_addon()->incoming_spike(timestamp, s, this, network);
                }
                
                // to handle case where the neuron never fires - for validation with the cadence experiments
                if (threshold != 0 && potential >= threshold) {
                    // save spikes on the LIF layer before the Decision Layer for classification purposes if there's a decision-making layer
                    if (network->get_decision_making() && network->get_decision_parameters().layer_number == layer_id+1) {
                        if (static_cast<int>(decision_queue.size()) < network->get_decision_parameters().spike_history_size) {
                            decision_queue.emplace_back(network->get_current_label());
                        } else {
                            decision_queue.pop_front();
                            decision_queue.emplace_back(network->get_current_label());
                        }
                    }
                    
                    if (network->get_verbose() == 2) {
                        std::cout << "t " << timestamp << " " << s->get_presynaptic_neuron_id() << "->" << neuron_id << " i_z " << current << " v_mem " << potential << " --> SPIKED" << std::endl;
                    }
                    
                    // propagate spike through the axon terminals
                    for (auto& axonTerminal : axon_terminals) {
                        auto& postsynaptic_layer = network->get_layers()[network->get_neurons()[axonTerminal->get_postsynaptic_neuron_id()]->get_layer_id()];
                        if (postsynaptic_layer.active) {
                            // dealing with feedforward and lateral connections
                            if (postsynaptic_layer.id >= layer_id) {
                                network->inject_spike(spike{timestamp + axonTerminal->get_delay(), axonTerminal.get(), spike_type::generated});
                            // dealing with feedback connections
                            } else {
                                auto& presynaptic_neuron = network->get_neurons()[axonTerminal->get_postsynaptic_neuron_id()];
                                // potentiation order flag STDP
                                if (potentiation_flag) {
                                    // send postsynaptic pulse after 13us
                                    network->inject_spike(spike{timestamp + 13, axonTerminal.get(), spike_type::trigger_down});
                                    network->inject_spike(spike{timestamp + 13 + membrane_time_constant, axonTerminal.get(), spike_type::trigger_down_to_up});
                                    network->inject_spike(spike{timestamp + 13 + tau_down_spike, axonTerminal.get(), spike_type::end_trigger_up});
                                    
                                    // if presynaptic neuron was active at some point
                                    if (presynaptic_neuron->get_trace() == 1) {
                                        // inject trigger_down spike to presynaptic_neuron to restart inference after 12us
                                        network->inject_spike(spike{timestamp + 12, axonTerminal.get(), spike_type::trigger_down});
                                        network->inject_spike(spike{timestamp + 12 + tau_down_event, axonTerminal.get(), spike_type::end_trigger_down});
                                        
                                    } else {
                                        // inject trigger_up spike to presynaptic_neuron for depression after 14us
                                        network->inject_spike(spike{timestamp + 14, axonTerminal.get(), spike_type::trigger_up});
                                        network->inject_spike(spike{timestamp + 14 + membrane_time_constant, axonTerminal.get(), spike_type::end_trigger_up});
                                    }
                                // depression inhibitor flag STDP
                                } else {
                                    // send postsynaptic pulse instantly and any currently spiking neurons will automatically potentiate
                                    network->inject_spike(spike{timestamp + 13, axonTerminal.get(), spike_type::trigger_down});
                                    network->inject_spike(spike{timestamp + 13 + membrane_time_constant, axonTerminal.get(), spike_type::trigger_down_to_up});
                                    network->inject_spike(spike{timestamp + 13 + tau_down_spike, axonTerminal.get(), spike_type::end_trigger_up});
                                    
                                    if (presynaptic_neuron->get_trace() == 0) {
                                        // inject trigger_up spike to presynaptic_neuron for depression after 1us
                                        network->inject_spike(spike{timestamp + 1, axonTerminal.get(), spike_type::trigger_up});
                                        network->inject_spike(spike{timestamp + 1 + membrane_time_constant, axonTerminal.get(), spike_type::end_trigger_up});
                                    }
                                }
                                
                                // reset trace
                                presynaptic_neuron->set_trace(0);
                            }
                        }
                    }
                    
                    // everytime a postsynaptic neuron fires increment refractory counter on all postsynaptic neurons that are currently inactive
                    check_refractory(network);
                    
                    // winner-takes-all to reset potential on all neurons in the same layer
                    winner_takes_all(timestamp, network);
                    
                    // disable neuron for refractory period
                    active = false;
                    
                    // save time when neuron fired
                    previous_spike_time = timestamp;
                }
                
                // save computation time
                previous_input_time = timestamp;
            }
		}
        
        // reset a neuron to its initial status
        virtual void reset_neuron(Network* network, bool clearAddons=true) override {
            previous_input_time = 0;
            previous_spike_time = 0;
            potential = resting_potential;
            refractory_counter = 0;
            trace = 0;

            for (auto& dendrite: dendritic_tree) {
                dendrite->reset();
            }

            for (auto& axon_terminal: axon_terminals) {
                axon_terminal->reset();
            }

            if (clearAddons) {
                relevant_addons.clear();
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
        
        virtual int share_information() override {
            refractory_counter += 1;
            return refractory_counter;
        }
        
    protected:
        
        void winner_takes_all(double timestamp, Network* network) override {
            for (auto& n: network->get_layers()[layer_id].neurons) {
                auto& neuron = network->get_neurons()[n];
                neuron->set_potential(resting_potential);
            }
        }
        
        // loops through any learning rules and activates them
        virtual void request_learning(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            if (network->get_learning_status()) {
                for (auto& addon: relevant_addons) {
                    addon->learn(timestamp, s, postsynapticNeuron, network);
                }
            }
        }
        
        void check_refractory(Network* network) {
            for (auto& n: network->get_layers()[layer_id].neurons) {
                auto& neuron = network->get_neurons()[n];
                if (!neuron->get_activity()) {
                    neuron->share_information();
                }
            }
        }
        
        double  epsilon;
        double  i_cancel;
        double  i_discharge;
        double  scaling_factor;
        bool    potentiation_flag;
        double  tau_down_event;
        double  tau_down_spike;
        int     refractory_counter;
	};
}
