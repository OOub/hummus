/*
 * CUBA_LIF.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 21/01/2019
 *
 * Information: current-based leaky integrate and fire (LIF) neuron model
 *
 * NEURON TYPE 1 (in JSON SAVE FILE)
 */

#pragma once

#include "../../third_party/json.hpp"

namespace hummus {
    
    class Synapse;
    class Neuron;
    class Network;
    
	class CUBA_LIF : public Neuron {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        CUBA_LIF(int _neuronID, int _layerID, int _sublayerID, int _rf_id,  std::pair<int, int> _xyCoordinates, int _refractoryPeriod=3, float _capacitance=200, float _leakageConductance=10, bool _homeostasis=false, bool _burstingActivity=false, float _traceTimeConstant=20, float _decayHomeostasis=20, float _homeostasisBeta=0.1, float _threshold=-50, float _restingPotential=-70, std::string _classLabel="") :
                Neuron(_neuronID, _layerID, _sublayerID, _rf_id, _xyCoordinates, _refractoryPeriod, _capacitance, _leakageConductance, _traceTimeConstant, _threshold, _restingPotential, _classLabel),
                bursting_activity(_burstingActivity),
                homeostasis(_homeostasis),
                resting_threshold(_threshold),
                decay_homeostasis(_decayHomeostasis),
                homeostasis_beta(_homeostasisBeta),
                active_synapse(nullptr) {
                    
            // LIF neuron type == 1 (for JSON save)
            neuron_type = 1;
                    
            inv_trace_tau = 1. / _traceTimeConstant;
            inv_membrane_tau = 1./ membrane_time_constant;
            inv_homeostasis_tau = 1. / _decayHomeostasis;
		}
		
		virtual ~CUBA_LIF(){}
		
		// ----- PUBLIC LIF METHODS -----        
		virtual void initialisation(Network* network) override {
            // searching for addons that are relevant to this neuron. if addons do not have a mask they are automatically not filtered out unless specified by the no_automatic_include() flag (tells the neuron not to automatically include the addon)
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
            
            // asynchronous network cannot use exponential synapses
            if (network->is_asynchronous()) {
                if (std::any_of(axon_terminals.begin(), axon_terminals.end(), [](std::unique_ptr<Synapse>& synapse) {
                    return dynamic_cast<Exponential*>(synapse.get()) != nullptr;
                })) {
                    throw std::logic_error("Exponential synapses are not compatible with the event-based mode");
                }
            }
		}
        
        // homeostasis does not work for the event-based neuron because it would complicate spike prediction
        virtual void update(double timestamp, Synapse* s, Network* network, float timestep, spike_type type) override {
                        
            float input_td = timestamp - previous_input_time;
            
            if (type == spike_type::initial || type == spike_type::generated) {
                
                // checking if the neuron is in a refractory period
                if (static_cast<float>(timestamp - previous_spike_time) >= refractory_period) {
                    active = true;
                }
				
                // updating current of synapses
                if (type == spike_type::initial) {
                    current = s->update(timestamp, timestep);
                } else {
                    float total_current = 0;
                    for (auto& synapse: dendritic_tree) {
                        total_current += synapse->update(timestamp, timestep);
                    }
                    current = total_current;
                }
                
                float exp_input_mem_tau = std::exp(- input_td * inv_membrane_tau);
                
                // trace decay
                trace -= trace * input_td * inv_trace_tau;
                
                // potential decay
                potential = resting_potential + (potential-resting_potential) * exp_input_mem_tau;
                
                if (network->get_main_thread_addon()) {
                    network->get_main_thread_addon()->status_update(timestamp, this, network);
                }
                
                if (active) {
                    
					// calculating the potential
                    potential = resting_potential + current * (1 - exp_input_mem_tau) + (potential - resting_potential) * exp_input_mem_tau;
                    
                    // sending spike to relevant synapse
                    s->receive_spike();
                    
                    if (type == spike_type::initial) {
                        current = s->get_synaptic_current();
                    } else {
                        // integating synaptic currents from dendritic tree
                        float total_current = 0;
                        for (auto& synapse: dendritic_tree) {
                            total_current += synapse->get_synaptic_current();
                        }
                        current = total_current;
                    }
                    
                    if (network->get_verbose() == 2) {
                        std::cout << "t=" << timestamp << " " << s->get_presynaptic_neuron_id() << "->" << neuron_id << " w=" << s->get_weight() << " d=" << s->get_delay() <<" V=" << potential << " Vth=" << threshold << " layer=" << layer_id << " --> EMITTED" << std::endl;
                    }
                    
                    for (auto& addon: relevant_addons) {
                        if (potential < threshold) {
                            addon->incoming_spike(timestamp, s, this, network);
                        }
                    }
                    
                    if (network->get_main_thread_addon()) {
                        network->get_main_thread_addon()->incoming_spike(timestamp, s, this, network);
                    }
					
                    if (current > 0) {
                        // calculating time at which potential = threshold
                        double predictedTimestamp = membrane_time_constant * (- std::log( - threshold + resting_potential + current) + std::log( current - potential + resting_potential)) + timestamp;
                        if (predictedTimestamp > timestamp && predictedTimestamp <= timestamp + s->get_synapse_time_constant()) {
                            network->inject_predicted_spike(spike{predictedTimestamp, s, spike_type::prediction}, spike_type::prediction);
                        } else {
                            network->inject_predicted_spike(spike{timestamp + s->get_synapse_time_constant(), s, spike_type::end_of_integration}, spike_type::end_of_integration);
                        }
                    } else {
                        potential = resting_potential + current * (1 - exp_input_mem_tau) + (potential - resting_potential);
                        network->remove_predicted_spike(neuron_id);
                    }
                }
            } else if (type == spike_type::prediction) {
                if (active) {
                    potential = resting_potential + current * (1 - std::exp(- input_td * inv_membrane_tau)) + (potential - resting_potential);
                }
            } else if (type == spike_type::end_of_integration) {
                if (active) {
                    float exp_s_tau_mem_tau = std::exp(-s->get_synapse_time_constant() * inv_membrane_tau);
                    potential = resting_potential + current * (1 - exp_s_tau_mem_tau) + (potential - resting_potential) * exp_s_tau_mem_tau;
                }
            }
        
            if (network->get_main_thread_addon()) {
                network->get_main_thread_addon()->status_update(timestamp, this, network);
            }
            
            if (potential >= threshold) {
                
                // save spikes on final LIF layer before the Decision Layer for classification purposes if there's a decision-making layer
                if (network->get_decision_making() && network->get_decision_parameters().layer_number == layer_id+1) {
                    if (decision_queue.size() < network->get_decision_parameters().spike_history_size) {
                        decision_queue.emplace_back(network->get_current_label());
                    } else {
                        decision_queue.pop_front();
                        decision_queue.emplace_back(network->get_current_label());
                    }
                }
                
                trace = 1;
                
                if (network->get_verbose() == 2) {
                    std::cout << "t=" << timestamp << " " << s->get_presynaptic_neuron_id() << "->" << neuron_id << " w=" << s->get_weight() << " d=" << s->get_delay() <<" V=" << potential << " Vth=" << threshold << " layer=" << layer_id << " --> SPIKED" << std::endl;
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
                
                // resetting the current after firing if we don't want the neuron to burst
                if (!bursting_activity) {
                    for (auto& synapse: dendritic_tree) {
                        synapse->reset();
                    }
                }
                
                if (network->get_main_thread_addon()) {
                    network->get_main_thread_addon()->status_update(timestamp, this, network);
                }
                
                previous_spike_time = timestamp;
                potential = resting_potential;
                active = false;
            }
            
            // updating the timestamp when a synapse was propagating a spike
            previous_input_time = timestamp;
            s->set_previous_input_time(timestamp);
		}
		
        virtual void update_sync(double timestamp, Synapse* s, Network* network, float timestep, spike_type type) override {
            
            // handling multiple spikes at the same timestamp (to prevent excessive decay)
            if (timestamp != 0 && timestamp - previous_input_time == 0) {
                timestep = 0;
            }
            
            // checking if the neuron is in a refractory period
            if (timestamp - previous_spike_time >= refractory_period) {
                active = true;
            }
            
            // updating current of synapses
            if (type == spike_type::initial) {
                current = s->update(timestamp, timestep);
            } else {
                float total_current = 0;
                for (auto& synapse: dendritic_tree) {
                    total_current += synapse->update(timestamp, timestep);
                }
                current = total_current;
            }
            
            // trace decay
            trace -= trace * timestep * inv_trace_tau;
            
			// potential decay
            potential += (resting_potential - potential) * timestep * inv_membrane_tau;
            
			// threshold decay
			if (homeostasis) {
                threshold += (resting_threshold - threshold) * timestep * inv_homeostasis_tau;
			}
                
			// neuron inactive during refractory period
			if (active) {
				if (s) {
                    
                    active_synapse = s;
                    
                    // updating the timestamp when a synapse was propagating a spike
                    previous_input_time = timestamp;
                    s->set_previous_input_time(timestamp);
                    
					// updating the threshold
					if (homeostasis) {
						threshold += homeostasis_beta * inv_homeostasis_tau;
					}
                    
                    // sending spike to relevant synapse
                    s->receive_spike();
                    
                    // integating synaptic currents
                    if (type == spike_type::initial) {
                        current = s->get_synaptic_current();
                    } else {
                        float total_current = 0;
                        for (auto& synapse: dendritic_tree) {
                            total_current += synapse->get_synaptic_current();
                        }
                        current = total_current;
                    }
                    
                    if (network->get_verbose() == 2) {
                        std::cout << "t=" << timestamp << " " << s->get_presynaptic_neuron_id() << "->" << neuron_id << " w=" << s->get_weight() << " d=" << s->get_delay() <<" V=" << potential << " Vth=" << threshold << " layer=" << layer_id << " --> EMITTED" << std::endl;
                    }
                    
                    for (auto& addon: relevant_addons) {
                        addon->incoming_spike(timestamp, s, this, network);
                    }
                    
                    if (network->get_main_thread_addon()) {
                        network->get_main_thread_addon()->incoming_spike(timestamp, s, this, network);
                    }
				}
				
                potential += current * (1 - std::exp(-timestep * inv_membrane_tau));
            }
            
            for (auto& addon: relevant_addons) {
                addon->status_update(timestamp, this, network);
            }
            
            if (network->get_main_thread_addon()) {
                network->get_main_thread_addon()->status_update(timestamp, this, network);
            }

			if (potential >= threshold && active_synapse) {
                
                // save spikes on final LIF layer before the Decision Layer for classification purposes if there's a decision-making layer
                if (network->get_decision_making() && network->get_decision_parameters().layer_number == layer_id+1) {
                    if (decision_queue.size() < network->get_decision_parameters().spike_history_size) {
                        decision_queue.emplace_back(network->get_current_label());
                    } else {
                        decision_queue.pop_front();
                        decision_queue.emplace_back(network->get_current_label());
                    }
                }
                
                trace += 1;
                
                if (network->get_verbose() == 2) {
                    std::cout << "t=" << timestamp << " " << active_synapse->get_presynaptic_neuron_id() << "->" << neuron_id << " w=" << active_synapse->get_weight() << " d=" << active_synapse->get_delay() <<" V=" << potential << " Vth=" << threshold << " layer=" << layer_id << " --> SPIKED" << std::endl;
                }
                
                for (auto& addon: relevant_addons) {
                    addon->neuron_fired(timestamp, active_synapse, this, network);
				}
                if (network->get_main_thread_addon()) {
                    network->get_main_thread_addon()->neuron_fired(timestamp, active_synapse, this, network);
				}
                
                for (auto& axonTerminal : axon_terminals) {
                    auto& post_synaptic_layer = network->get_layers()[network->get_neurons()[axonTerminal->get_postsynaptic_neuron_id()]->get_layer_id()];
                    if (post_synaptic_layer.active) {
                        network->inject_spike(spike{timestamp + axonTerminal->get_delay(), axonTerminal.get(), spike_type::generated});
                    }
                }

                request_learning(timestamp, active_synapse, this, network);
                
				if (!bursting_activity) {
                    for (auto& synapse: dendritic_tree) {
                        synapse->reset();
                    }
				}
                
                
                previous_spike_time = timestamp;
                potential = resting_potential;
				active = false;
			}
		}
		
        virtual void reset_neuron(Network* network, bool clearAddons=true) override {
            previous_input_time = 0;
            previous_spike_time = 0;
            current = 0;
            potential = resting_potential;
            trace = 0;
            active = true;
            
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
                {"capacitance", capacitance},
                {"leakage_conductance", leakage_conductance},
                {"bursting_activity", bursting_activity},
                {"homeostasis", homeostasis},
                {"resting_threshold", resting_threshold},
                {"decay_homeostasis", decay_homeostasis},
                {"homeostasis_beta", homeostasis_beta},
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
        
		// ----- SETTERS AND GETTERS -----        
        void set_bursting_activity(bool new_bool) {
            bursting_activity = new_bool;
        }
        
        void set_homeostasis(bool new_bool) {
            homeostasis = new_bool;
        }
        
        void set_resting_threshold(float new_thres) {
            resting_threshold = new_thres;
        }
        
        void set_decay_homeostasis(float new_DH) {
            decay_homeostasis = new_DH;
        }
        
        void set_homeostasis_beta(float new_HB) {
            homeostasis_beta = new_HB;
        }
        
	protected:
		
        // loops through any learning rules and activates them
        virtual void request_learning(double timestamp, Synapse* s, Neuron* postsynaptic_neuron, Network* network) override {
            if (network->get_learning_status() && !relevant_addons.empty()) {
                for (auto& addon: relevant_addons) {
                    addon->learn(timestamp, s, postsynaptic_neuron, network);
                }
            }
        }
        
		// ----- LIF PARAMETERS -----
		bool                         active;
		bool                         bursting_activity;
		bool                         homeostasis;
		double                       resting_threshold;
		double                       decay_homeostasis;
		double                       homeostasis_beta;
		Synapse*                     active_synapse;
        
        // Parameters for performance improvement
        double                       inv_trace_tau;
        double                       inv_membrane_tau;
        double                       inv_homeostasis_tau;
	};
}
