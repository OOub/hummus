/*
 * ulpec_lif.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/01/2019
 *
 * Information: neuron modeled according to the ULPEC analog neuron made by the IMS at Université de Bordeaux.
 */

#pragma once

namespace hummus {
    
    class Synapse;
    class Neuron;
    class Network;
    
	class ULPEC_LIF : public Neuron {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        ULPEC_LIF(int _neuronID, int _layerID, int _sublayerID, int _rf_id,  std::pair<int, int> _xyCoordinates, int _refractoryPeriod=10, float _capacitance=5e-12, float _threshold=1.2, float _restingPotential=0, float _i_discharge=12e-9, float _epsilon=0, float _scaling_factor=650, bool _potentiation_flag=true, float _tau_up=0.5, float _tau_down_event=10, float _tau_down_spike=1.5, float _delta_v=1, bool _skip_after_post=false) :
                Neuron(_neuronID, _layerID, _sublayerID, _rf_id, _xyCoordinates, _refractoryPeriod, _capacitance, 0, 0, _threshold, _restingPotential, -1),
                epsilon(_epsilon),
                i_discharge(_i_discharge),
                scaling_factor(_scaling_factor),
                potentiation_flag(_potentiation_flag),
                tau_up(_tau_up),
                tau_down_event(_tau_down_event),
                tau_down_spike(_tau_down_spike),
                refractory_counter(0),
                delta_v(_delta_v),
                skip_after_post(_skip_after_post) {
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
        
        virtual void update(double timestamp, Synapse* s, Network* network, float timestep, spike_type type) override {
            // during testing there is no refractory period
            if (!network->get_learning_status() && refractory_period != 0) {
                refractory_period = 0;
            }
            
            // checking whether a refractory period is over
            if (!active && refractory_counter >= refractory_period) {
                active = true;
                refractory_counter = 0;
            }
            
            if ((type == spike_type::initial && active) || (type == spike_type::generated && active)) {
                // compute the current
                compute_current();
                
                // compute the potential V_membrane
                compute_potential(timestamp, s, network);
                
                // injecting potential in the memristor
                s->receive_spike(-delta_v);
                
                // check if the threshold is crossed, and start the corresponding chain of events if it is
                threshold_cross_check(timestamp, s, network);
                
                // sending a spike via the same synapse to signal the end of inference
                network->inject_spike(spike{timestamp + tau_down_event, s, spike_type::end_of_integration});
                
            } else if (type == spike_type::end_of_integration) {
                
                // compute the current
                compute_current();

                // compute the potential V_membrane
                compute_potential(timestamp, s, network);

                // check if the threshold is crossed, and start the corresponding chain of events if it is
                threshold_cross_check(timestamp, s, network);

                // update the GUI just before inference ends
                auto& presynaptic_neuron = network->get_neurons()[s->get_presynaptic_neuron_id()];
                if (network->get_main_thread_addon()) {
                    network->get_main_thread_addon()->status_update(timestamp, presynaptic_neuron.get(), network);
                }
                
                // removing injected potential from the memristor
                s->reset();
                
                // remove the inject potential from the presynaptic neuron
                presynaptic_neuron->set_potential(presynaptic_neuron->get_resting_potential());
                
                // update the GUI just after inference ends
                if (network->get_main_thread_addon()) {
                    network->get_main_thread_addon()->status_update(timestamp, presynaptic_neuron.get(), network);
                }
            } else if (type == spike_type::trigger_down) {
                
                // injecting potential in the memristor
                s->receive_spike(-delta_v);

            } else if (type == spike_type::trigger_up) {
                
                // injecting potential in the memristor
                s->receive_spike(delta_v);
                
            } else if (type == spike_type::trigger_down_to_up) {
                
                // from down to up to get the postsynaptic waveform
                s->receive_spike(2*delta_v);

            } else if (type == spike_type::end_trigger_up) {
                for (auto& addon: relevant_addons) {
                    addon->incoming_spike(timestamp, s, this, network);
                }
                
                // remove the injected potential in the memristor
                s->receive_spike(-delta_v);
                
            } else if (type == spike_type::end_trigger_down) {
                
                for (auto& addon: relevant_addons) {
                    addon->incoming_spike(timestamp, s, this, network);
                }
                
                // remove the injected potential in the memristor
                s->receive_spike(delta_v);
                
            }
            
            // activate learning rule whenever learning threshold is passed
            request_learning(timestamp, s, this, network);
            
            // save computation time
            previous_input_time = timestamp;
        }
        
        // reset a neuron to its initial status
        virtual void reset_neuron(Network* network, bool clearAddons=true) override {
            previous_input_time = 0;
            previous_spike_time = 0;
            potential = resting_potential;
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
        
    protected:
        
        // computing the neuron's current
        void compute_current() {
            // compute r_network
            float r_network = 0;
            for (auto& memristor: dendritic_tree) {
                // taking the inactive memristors
                if (memristor->get_synaptic_current() <= 0) {
                    r_network = 1. / memristor->get_weight();
                }
            }
            
            // compute i_cancel
            float i_cancel = 0;
            if (r_network > 0) {
                 i_cancel = epsilon / r_network;
            }
            
            // getting the current i_x
            float i_x = 0;
            for (auto& memristor: dendritic_tree) {
                // taking only the active memristors
                if (memristor->get_synaptic_current() > 0) {
                    i_x += memristor->get_synaptic_current();
                }
            }
            
            // calculating current (i_z) taking into consideration the scaling factor
            if (i_x > i_cancel) {
                current = (i_x - i_cancel) / scaling_factor;
            } else {
                current = 0;
            }
        }
        
        // computing the neuron's V_membrane
        void compute_potential(double timestamp, Synapse* s, Network* network) {
            float delta_t = static_cast<float>((timestamp - previous_input_time) * 1e-6); /// delta_t converted to seconds
            potential += (current * delta_t / capacitance) - (i_discharge * delta_t / capacitance);
            
            if (potential < 0) {
                potential = 0;
            }
            
            if (network->get_verbose() == 2 && potential < threshold) {
                std::cout << "t " << timestamp << " " << s->get_presynaptic_neuron_id() << "->" << neuron_id << " i_z " << current << " v_mem " << potential << " layer id " << layer_id << std::endl;
            }

            for (auto& addon: relevant_addons) {
                addon->incoming_spike(timestamp, s, this, network);
            }

            if (network->get_main_thread_addon()) {
                network->get_main_thread_addon()->incoming_spike(timestamp, s, this, network);
            }
        }
        
        // check if the threshold is crossed, and start the corresponding chain of events if it is
        void threshold_cross_check(double timestamp, Synapse* s, Network* network) {
            if (threshold != 0 && potential >= threshold) {
                
                // save spikes on the LIF layer before the Decision Layer for classification purposes if there's a decision-making layer
                if (network->get_learning_status() && network->get_decision_making() && network->get_decision_parameters().layer_number == layer_id+1) {
                    if (static_cast<int>(decision_queue.size()) < network->get_decision_parameters().spike_history_size) {
                        decision_queue.emplace_back(network->get_current_label());
                    } else {
                        decision_queue.pop_front();
                        decision_queue.emplace_back(network->get_current_label());
                    }
                }
                
                if (network->get_verbose() == 2) {
                    std::cout << "t " << timestamp << " " << s->get_presynaptic_neuron_id() << "->" << neuron_id << " i_z " << current << " v_mem " << potential << " layer id " << layer_id << " --> SPIKED" << std::endl;
                }
                
                for (auto& addon: relevant_addons) {
                    addon->neuron_fired(timestamp, s, this, network);
                }
                
                if (network->get_main_thread_addon()) {
                    network->get_main_thread_addon()->neuron_fired(timestamp, s, this, network);
                }
                
                // propagate spike through the axon terminals (towards decision-making neurons)
                for (auto& axon_terminal : axon_terminals) {
                    auto& postsynaptic_layer = network->get_layers()[network->get_neurons()[axon_terminal->get_postsynaptic_neuron_id()]->get_layer_id()];
                    if (postsynaptic_layer.active) {
                        network->inject_spike(spike{timestamp, axon_terminal.get(), spike_type::generated});
                    }
                }
                
                for (auto& dendrite: dendritic_tree) {

                    auto& presynaptic_neuron = network->get_neurons()[dendrite->get_presynaptic_neuron_id()];

                    // only apply programming pulses during the learning phase
                    if (network->get_learning_status()) {
                        // POF learning pulse
                        if (potentiation_flag) {
//                            // send postsynaptic pulse after 13us
//                            network->inject_spike(spike{timestamp + 13, dendrite, spike_type::trigger_down});
//                            network->inject_spike(spike{timestamp + 13 + tau_up, dendrite, spike_type::trigger_down_to_up});
//                            network->inject_spike(spike{timestamp + 13 + tau_up + tau_down_spike, dendrite, spike_type::end_trigger_up});
//
//                            // TESTING FOR EFFECT OF TIME
//                            if (presynaptic_neuron->get_trace() == 1 && timestamp - presynaptic_neuron->get_previous_spike_time() <= 100) {
//                                // inject trigger_down spike to presynaptic_neuron to restart inference after 12us
//                                network->inject_spike(spike{timestamp + 12, dendrite, spike_type::trigger_down});
//                                network->inject_spike(spike{timestamp + 12 + tau_down_event, dendrite, spike_type::end_trigger_down});
//
//                            } else if (presynaptic_neuron->get_trace() == 0) {
//                                // inject trigger_up spike to presynaptic_neuron for depression after 14us
//                                network->inject_spike(spike{timestamp + 14, dendrite, spike_type::trigger_up});
//                                network->inject_spike(spike{timestamp + 14 + tau_up, dendrite, spike_type::end_trigger_up});
//                            }
                            
                            // if presynaptic neuron was active at some point
                            if (presynaptic_neuron->get_trace() == 1) {
                                // inject trigger_down spike to presynaptic_neuron to restart inference after 12us
                                network->inject_spike(spike{timestamp + 12, dendrite, spike_type::trigger_down});
                                network->inject_spike(spike{timestamp + 12 + tau_down_event, dendrite, spike_type::end_trigger_down});

                            } else {
                                // inject trigger_up spike to presynaptic_neuron for depression after 14us
                                network->inject_spike(spike{timestamp + 14, dendrite, spike_type::trigger_up});
                                network->inject_spike(spike{timestamp + 14 + tau_up, dendrite, spike_type::end_trigger_up});
                            }
                            
                        // DIF learning pulse
                        } else {
                            // send postsynaptic pulse instantly and any currently spiking neurons will automatically potentiate
                            network->inject_spike(spike{timestamp, dendrite, spike_type::trigger_down});
                            network->inject_spike(spike{timestamp + tau_up, dendrite, spike_type::trigger_down_to_up});
                            network->inject_spike(spike{timestamp + tau_up + tau_down_spike, dendrite, spike_type::end_trigger_up});

                            if (presynaptic_neuron->get_trace() == 0) {
                                // inject trigger_up spike to presynaptic_neuron for depression after 1us
                                network->inject_spike(spike{timestamp + 1, dendrite, spike_type::trigger_up});
                                network->inject_spike(spike{timestamp + 1 + tau_up, dendrite, spike_type::end_trigger_up});
                            }
                        }
                    }
                    // reset trace
                    presynaptic_neuron->set_trace(0);
                }
                
                // everytime a postsynaptic neuron fires increment refractory counter on all postsynaptic neurons that are currently inactive
                check_refractory(network);

                // winner-takes-all to reset potential on all neurons in the same layer
                winner_takes_all(timestamp, network);

                // disable neuron for refractory period
                active = false;

                // save time when neuron fired
                previous_spike_time = timestamp;
                
                // skip presentation after learning is completed
                if (skip_after_post && network->get_learning_status()) {
                    if (potentiation_flag) {
                        network->set_skip_presentation(timestamp + 25);
                    } else {
                        network->set_skip_presentation(timestamp + 13);
                    }
                }
            }
        }
        
        void winner_takes_all(double timestamp, Network* network) override {
            for (auto& n: network->get_layers()[layer_id].neurons) {
                auto& neuron = network->get_neurons()[n];
                neuron->set_potential(resting_potential);
            }
        }
        
        // loops through any learning rules and activates them
        virtual void request_learning(double timestamp, Synapse* s, Neuron* postsynapticNeuron, Network* network) override {
            if (network->get_learning_status() && !relevant_addons.empty()) {
                for (auto& addon: relevant_addons) {
                    addon->learn(timestamp, s, postsynapticNeuron, network);
                }
            }
        }
        
        void check_refractory(Network* network) {
            if (refractory_period > 0) {
                for (auto& n: network->get_layers()[layer_id].neurons) {
                    auto& neuron = network->get_neurons()[n];
                    if (neuron->get_neuron_id() != neuron_id && !neuron->get_activity()) {
                        dynamic_cast<ULPEC_LIF*>(neuron.get())->increment_refractory_counter();
                    }
                }
            }
        }
        
        void increment_refractory_counter() {
            refractory_counter++;
        }
        
        float  epsilon;
        float  i_discharge;
        float  scaling_factor;
        bool   potentiation_flag;
        float  tau_up;
        float  tau_down_event;
        float  tau_down_spike;
        int    refractory_counter;
        float  delta_v;
        bool   skip_after_post;
	};
}
