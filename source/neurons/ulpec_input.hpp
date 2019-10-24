/*
 * ulpec_input.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/01/2019
 *
 * Information: ULPEC_Input neurons generates waveforms used in the context of the ulpec project.
 *
 * NEURON TYPE 4 (in JSON SAVE FILE)
 */

#pragma once

#include "../../third_party/json.hpp"

namespace hummus {

    class Synapse;
    class Neuron;
    class Network;

    class ULPEC_Input : public Neuron {

    public:
        // ----- CONSTRUCTOR AND DESTRUCTOR -----
        ULPEC_Input(int _neuronID, int _layerID, int _sublayerID, int _rf_id,  std::pair<int, int> _xyCoordinates, int _refractoryPeriod=25, double _threshold=1.2, double _restingPotential=1.1, double _tau=10, double _input_voltage=1, bool _positive_waveform=false) :
                Neuron(_neuronID, _layerID, _sublayerID, _rf_id, _xyCoordinates, _refractoryPeriod, 0, 0, 0, _threshold, _restingPotential, ""),
                input_voltage(_input_voltage),
                positive_waveform(_positive_waveform) {

            // DecisionMaking neuron type = 2 for JSON save
            neuron_type = 4;
            membrane_time_constant = _tau;
        }

        virtual ~ULPEC_Input(){}

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
            
            if (network->get_main_thread_addon()) {
                network->get_main_thread_addon()->status_update(timestamp, this, network);
            }
            
            // checking if the neuron is in a refractory period
            if (timestamp - previous_spike_time >= refractory_period) {
                active = true;
            }
            
            // initial spike = AER event
            if (type == spike_type::initial && active) {
                if (positive_waveform) {
                    // updating neuron potential
                    potential += input_voltage;
                    
                    // end of integration spike on the same neuron after tau_down_event
                    network->inject_spike(spike{timestamp + membrane_time_constant, s, spike_type::end_of_integration});
                    
                    // inject voltage into the memristors and propagate spike to the next layer
                    for (auto& axon_terminal: axon_terminals) {
                        axon_terminal->receive_spike(input_voltage);
                        network->inject_spike(spike{timestamp, axon_terminal.get(), spike_type::generated});
                    }
                } else {
                    // updating neuron potential
                    potential -= input_voltage;
                    
                    // end of integration spike on the same neuron after tau_down_event
                    network->inject_spike(spike{timestamp + membrane_time_constant, s, spike_type::end_of_integration});
                    
                    // inject voltage into the memristors and propagate spike to the next layer
                    for (auto& axon_terminal: axon_terminals) {
                        axon_terminal->receive_spike(-input_voltage);
                        network->inject_spike(spike{timestamp, axon_terminal.get(), spike_type::generated});
                    }
                }
                
                if (network->get_verbose() == 2) {
                    std::cout << "t=" << timestamp << " " << neuron_id << " --> INFERENCE" << " V=" << potential << std::endl;
                }
                
                // send neuron_fired signal to the addons
                for (auto& addon: relevant_addons) {
                    addon->neuron_fired(timestamp, s, this, network);
                }
                
                if (network->get_main_thread_addon()) {
                    network->get_main_thread_addon()->neuron_fired(timestamp, s, this, network);
                }
                
                // set trace to 1
                trace = 1;
                
                // setting time when inference started
                previous_spike_time = timestamp;
                
                // starting refractory period to accept AER events
                active = false;
                
            // spike generated by the postsynaptic neuron
            } else if (type == spike_type::trigger_down) {
//                // updating neuron potential
//                potential -= input_voltage;
//
//                // inject voltage into the memristors connected to the postsynaptic neuron
//                for (auto& axon_terminal: axon_terminals) {
//                    if (axon_terminal->get_postsynaptic_neuron_id() == s->get_presynaptic_neuron_id()) {
//                        axon_terminal->receive_spike(-input_voltage);
//                    }
//                }
                
            // spike generated by the postsynaptic neuron
            } else if (type == spike_type::trigger_up) {
//                // updating neuron potential
//                potential += input_voltage;
//
//                // inject voltage into the memristors
//                for (auto& axon_terminal: axon_terminals) {
//                    if (axon_terminal->get_postsynaptic_neuron_id() == s->get_presynaptic_neuron_id()) {
//                        axon_terminal->receive_spike(input_voltage);
//                    }
//                }
            
            } else if (type == spike_type::trigger_down_to_up) {
//            // updating neuron potential
//            potential += 2 * input_voltage;
//
//            // inject voltage into the memristors
//            for (auto& axon_terminal: axon_terminals) {
//                if (axon_terminal->get_postsynaptic_neuron_id() == s->get_presynaptic_neuron_id()) {
//                    axon_terminal->receive_spike(2 * input_voltage);
//                }
//            }
                
            // end of the square waveform generated by the input AER event
            } else if (type == spike_type::end_of_integration) {
                if (positive_waveform) {
                    // removing the contribution of the input event on the neuron potential
                    potential -= input_voltage;
                    
                    // also removing the contribution on the synaptic potential
                    for (auto& axon_terminal: axon_terminals) {
                        axon_terminal->receive_spike(-input_voltage);
                        network->inject_spike(spike{timestamp, axon_terminal.get(), spike_type::generated});
                    }
                } else {
                    // removing the contribution of the input event on the neuron potential
                    potential += input_voltage;
                    
                    // also removing the contribution on the synaptic potential
                    for (auto& axon_terminal: axon_terminals) {
                        axon_terminal->receive_spike(input_voltage);
                    }
                }
                
                if (network->get_verbose() == 2) {
                    std::cout << "t=" << timestamp << " " << neuron_id << " --> END OF INFERENCE" << " V=" << potential << std::endl;
                }
                
            // end of the square waveform generated by the trigger_up event
            } else if (type == spike_type::end_trigger_up) {
//                // removing the contribution of the trigger_up event on the neuron potential
//                potential -= input_voltage;
//
//                // also removing the contribution on the synaptic potential
//                for (auto& axon_terminal: axon_terminals) {
//                    if (axon_terminal->get_postsynaptic_neuron_id() == s->get_presynaptic_neuron_id()) {
//                        axon_terminal->receive_spike(-input_voltage);
//                    }
//                }
            // end of the square waveform generated by the trigger_down event
            } else if (type == spike_type::end_trigger_down) {
//                // removing the contribution of the trigger_up event on the neuron potential
//                potential += input_voltage;
//
//                // also removing the contribution on the synaptic potential
//                for (auto& axon_terminal: axon_terminals) {
//                    if (axon_terminal->get_postsynaptic_neuron_id() == s->get_presynaptic_neuron_id()) {
//                        axon_terminal->receive_spike(input_voltage);
//                    }
//                }
            }

            for (auto& axon_terminal: axon_terminals) {
                request_learning(timestamp, axon_terminal.get(), this, network);
            }
            
            if (network->get_main_thread_addon()) {
                network->get_main_thread_addon()->status_update(timestamp, this, network);
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
                {"threshold", threshold},
                {"resting_potential", resting_potential},
                {"refractory_period", refractory_period},
                {"input_voltage", input_voltage},
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
        virtual void request_learning(double timestamp, Synapse* s, Neuron* postsynaptic_neuron, Network* network) override {
            if (network->get_learning_status() && !relevant_addons.empty()) {
                for (auto& addon: relevant_addons) {
                    addon->learn(timestamp, s, postsynaptic_neuron, network);
                }
            }
        }
        
        // ----- PULSE_GENERATOR PARAMETERS -----
        double       input_voltage;
        bool         positive_waveform;
    };
}
