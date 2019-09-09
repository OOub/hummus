/*
 * LIF.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 21/01/2019
 *
 * Information: leaky integrate and fire (LIF) neuron model with current dynamics.
 *
 * NEURON TYPE 1 (in JSON SAVE FILE)
 */

#pragma once

#include "../../third_party/json.hpp"

namespace hummus {
    
    class Synapse;
    class Neuron;
    class Network;
    
	class LIF : public Neuron {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        LIF(int _neuronID, int _layerID, int _sublayerID, std::pair<int, int> _rfCoordinates,  std::pair<float, float> _xyCoordinates, int _refractoryPeriod=3, float _conductance=200, float _leakageConductance=10, bool _homeostasis=false, bool _burstingActivity=false, float _traceTimeConstant=20, float _decayHomeostasis=20, float _homeostasisBeta=0.1, float _threshold=-50, float _restingPotential=-70, std::string _classLabel="") :
                Neuron(_neuronID, _layerID, _sublayerID, _rfCoordinates, _xyCoordinates, _refractoryPeriod, _conductance, _leakageConductance, _traceTimeConstant, _threshold, _restingPotential, _classLabel),
                active(true),
                bursting_activity(_burstingActivity),
                homeostasis(_homeostasis),
                resting_threshold(_threshold),
                decay_homeostasis(_decayHomeostasis),
                homeostasis_beta(_homeostasisBeta),
                active_synapse(nullptr),
                inhibited(false),
                inhibition_time(0){
                    
            // LIF neuron type == 1 (for JSON save)
                    neuron_type = 1;
		}
		
		virtual ~LIF(){}
		
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
            if (network->get_network_type()) {
                if (std::any_of(axon_terminals.begin(), axon_terminals.end(), [](std::unique_ptr<Synapse>& synapse) {
                    return dynamic_cast<Exponential*>(synapse.get()) != nullptr;
                })) {
                    throw std::logic_error("Exponential synapses are not compatible with the event-based mode");
                }
            }
		}
        
        // homeostasis does not work for the event-based neuron because it would complicate spike prediction
        virtual void update(double timestamp, Synapse* s, Network* network, spike_type type) override {
            
            if (type == spike_type::initial || type == spike_type::generated) {
                // checking if the neuron is inhibited
                if (inhibited && timestamp - inhibition_time >= refractory_period) {
                    inhibited = false;
                }
                
                // checking if the neuron is in a refractory period
                if (timestamp - previous_spike_time >= refractory_period) {
                    active = true;
                }
				
                // updating current of synapses
                if (type == spike_type::initial) {
                    current = s->update(timestamp);
                } else {
                    float total_current = 0;
                    for (auto& synapse: dendritic_tree) {
                        total_current += synapse->update(timestamp);
                    }
                    current = total_current;
                }
                
                // trace decay
                trace *= std::exp(-(timestamp-previous_input_time)/trace_time_constant);
                
                // potential decay
                potential = resting_potential + (potential-resting_potential)*std::exp(-(timestamp-previous_input_time)/membrane_time_constant);
                
                if (network->get_main_thread_addon()) {
                    network->get_main_thread_addon()->status_update(timestamp, s, this, network);
                }
                
                if (active && !inhibited) {
					// calculating the potential
                    potential = resting_potential + current * (1 - std::exp(-(timestamp-previous_input_time)/membrane_time_constant)) + (potential - resting_potential) * std::exp(-(timestamp-previous_input_time)/membrane_time_constant);
					
                    // sending spike to relevant synapse
                    s->receive_spike(timestamp);
                    
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
					
                    if (s->get_weight() >= 0) {
                        // calculating time at which potential = threshold
                        double predictedTimestamp = membrane_time_constant * (- std::log( - threshold + resting_potential + current) + std::log( current - potential + resting_potential)) + timestamp;
                        
                        if (predictedTimestamp > timestamp && predictedTimestamp <= timestamp + s->get_synapse_time_constant()) {
                            network->inject_predicted_spike(spike{predictedTimestamp, s, spike_type::prediction}, spike_type::prediction);
                        } else {
                            network->inject_predicted_spike(spike{timestamp + s->get_synapse_time_constant(), s, spike_type::end_of_integration}, spike_type::end_of_integration);
                        }
                    } else {
                        potential = resting_potential + current * (1 - std::exp(-(timestamp-previous_input_time)/membrane_time_constant)) + (potential - resting_potential);
                    }
                }
            } else if (type == spike_type::prediction) {
                if (active && !inhibited) {
                    potential = resting_potential + current * (1 - std::exp(-(timestamp-previous_input_time)/membrane_time_constant)) + (potential - resting_potential);
                }
            } else if (type == spike_type::end_of_integration) {
                if (active && !inhibited) {
                    potential = resting_potential + current * (1 - std::exp(-s->get_synapse_time_constant()/membrane_time_constant)) + (potential - resting_potential) * std::exp(-s->get_synapse_time_constant()/membrane_time_constant);
                }
            }
        
            if (network->get_main_thread_addon()) {
                network->get_main_thread_addon()->status_update(timestamp, s, this, network);
            }

            if (potential >= threshold) {
                // save spikes on final LIF layer before the Decision Layer for classification purposes if there's a decision-making layer
                if (network->get_decision_making() && network->getDecisionParameters().layer_number == layer_id+1) {
                    if (decision_queue.size() < network->getDecisionParameters().spike_history_size) {
                        decision_queue.emplace_back(network->get_current_label());
                    } else {
                        decision_queue.pop_front();
                        decision_queue.emplace_back(network->get_current_label());
                    }
                }
                
                trace += 1;

                if (network->get_verbose() == 2) {
                    std::cout << "t=" << timestamp << " " << s->get_presynaptic_neuron_id() << "->" << neuron_id << " w=" << s->get_weight() << " d=" << s->get_delay() <<" V=" << potential << " Vth=" << threshold << " layer=" << layer_id << " --> SPIKED" << std::endl;
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
                if (!bursting_activity) {
                    current = 0;
                    for (auto& synapse: dendritic_tree) {
                        synapse->reset();
                    }
                }
                active = false;
                
                if (network->get_main_thread_addon()) {
                    network->get_main_thread_addon()->status_update(timestamp, s, this, network);
                }
            }
            
            // updating the timestamp when a synapse was propagating a spike
            previous_input_time = timestamp;
            s->set_previous_input_time(timestamp);
		}
		
        virtual void update_sync(double timestamp, Synapse* s, Network* network, double timestep, spike_type type) override {
            // handling multiple spikes at the same timestamp (to prevent excessive decay)
            if (timestamp != 0 && timestamp - previous_spike_time == 0) {
                timestep = 0;
            }
            
            // checking if the neuron is inhibited
            if (inhibited && timestamp - inhibition_time >= refractory_period) {
				inhibited = false;
			}
            
            // checking if the neuron is in a refractory period
            if (timestamp - previous_spike_time >= refractory_period) {
                active = true;
            }
            
            // updating current of synapses
            if (type == spike_type::initial) {
                current = s->update(timestamp);
            } else {
                float total_current = 0;
                for (auto& synapse: dendritic_tree) {
                    total_current += synapse->update(timestamp);
                }
                current = total_current;
            }
            
            // trace decay
            trace *= std::exp(-timestep/trace_time_constant);
            
			// potential decay
            potential = resting_potential + (potential-resting_potential)*std::exp(-timestep/membrane_time_constant);
            
			// threshold decay
			if (homeostasis) {
                threshold = resting_threshold + (threshold-resting_threshold)*std::exp(-timestep/decay_homeostasis);
			}
                
			// neuron inactive during refractory period
			if (active && !inhibited) {
				if (s) {
                                        
					// updating the threshold
					if (homeostasis) {
						threshold += homeostasis_beta/decay_homeostasis;
					}
                    
                    // sending spike to relevant synapse
                    s->receive_spike(timestamp);
                    
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
					active_synapse = s;
                    
                    // updating the timestamp when a synapse was propagating a spike
                    previous_input_time = timestamp;
                    s->set_previous_input_time(timestamp);
                    
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
				}
				
                potential += current * (1 - std::exp(-timestep/membrane_time_constant));
            }
            
            if (s) {
                if (network->get_main_thread_addon()) {
                    network->get_main_thread_addon()->status_update(timestamp, s, this, network);
                }
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

			if (potential >= threshold && active_synapse) {
                
                // save spikes on final LIF layer before the Decision Layer for classification purposes if there's a decision-making layer
                if (network->get_decision_making() && network->getDecisionParameters().layer_number == layer_id+1) {
                    if (decision_queue.size() < network->getDecisionParameters().spike_history_size) {
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
                
                if (!bursting_activity) {
                    for (auto& synapse: dendritic_tree) {
                        synapse->reset();
                    }
                }
                
                for (auto& addon: relevant_addons) {
                    addon->neuron_fired(timestamp, active_synapse, this, network);
				}
                if (network->get_main_thread_addon()) {
                    network->get_main_thread_addon()->neuron_fired(timestamp, active_synapse, this, network);
				}
                
                if (!network->get_layers()[layer_id].do_not_propagate) {
                    for (auto& axonTerminal: axon_terminals) {
                        network->inject_spike(spike{timestamp + axonTerminal->get_delay(), axonTerminal.get(), spike_type::generated});
                    }
                }

                request_learning(timestamp, active_synapse, this, network);

                previous_spike_time = timestamp;
                potential = resting_potential;
                
				if (!bursting_activity) {
                    current = 0;
                    for (auto& synapse: dendritic_tree) {
                        if (synapse->get_weight() > 0) {
                            synapse->reset();
                        }
                    }
				}
                
				active = false;
			}
		}
		
        virtual void reset_neuron(Network* network, bool clearAddons=true) override {
            // resetting parameters
            previous_input_time = 0;
            previous_spike_time = 0;
            current = 0;
            potential = resting_potential;
            trace = 0;
            inhibited = false;
            active = true;
            threshold = resting_threshold;
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
                {"rf_coordinates", rf_coordinates},
                {"xy_coordinates", xy_coordinates},
                {"trace_time_constant", trace_time_constant},
                {"threshold", threshold},
                {"resting_potential", resting_potential},
                {"refractory_period", refractory_period},
                {"conductance", conductance},
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
		bool get_activity() const {
			return active;
		}
		
		void set_inhibition(double timestamp, bool inhibition_status) {
			inhibition_time = timestamp;
			inhibited = inhibition_status;
		}
        
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
            if (network->get_learning_status()) {
                if (!relevant_addons.empty()) {
                    for (auto& addon: relevant_addons) {
                        addon->learn(timestamp, s, postsynaptic_neuron, network);
                    }
                }
            }
        }
        
		// ----- LIF PARAMETERS -----
		bool                         active;
		bool                         inhibited;
		double                       inhibition_time;
		bool                         bursting_activity;
		bool                         homeostasis;
		float                        resting_threshold;
		float                        decay_homeostasis;
		float                        homeostasis_beta;
		Synapse*                     active_synapse;
	};
}
