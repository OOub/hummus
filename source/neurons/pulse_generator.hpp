/*
 * pulse_generator.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/01/2019
 *
 * Information: pulse_generator neurons generates waveforms used in the context of the ulpec project.
 *
 * NEURON TYPE 4 (in JSON SAVE FILE)
 */

#pragma once

#include "../../third_party/json.hpp"

namespace hummus {

    class Synapse;
    class Neuron;
    class Network;

    class Pulse_Generator : public Neuron {

    public:
        // ----- CONSTRUCTOR AND DESTRUCTOR -----
        Pulse_Generator(int _neuronID, int _layerID, int _sublayerID, int _rf_id,  std::pair<int, int> _xyCoordinates, int _refractoryPeriod=0, float _threshold=1.2, float _restingPotential=1.1, double _tau_up=0.5, double _tau_down=10, float _input_voltage=1) :
                Neuron(_neuronID, _layerID, _sublayerID, _rf_id, _xyCoordinates, _refractoryPeriod, 0, 0, 0, _threshold, _restingPotential, ""),
                tau_up(_tau_up),
                tau_down(_tau_down),
                input_voltage(_input_voltage) {

            // DecisionMaking neuron type = 2 for JSON save
            neuron_type = 4;
        }

        virtual ~Pulse_Generator(){}

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
            // initial spike = AER event
            if (type == spike_type::initial) {
                // 1. decrease in voltage square waveform (to 0.1V)
                potential -= input_voltage;
                
                // 2. end of integration spike on the same neuron after tau_down
                network->inject_spike(spike{timestamp + tau_down, s, spike_type::end_of_integration});
                
                // 3. set trace to 1 -> equivalent to the 1bit activity flag
                trace = 1;
                
                // 4. inject voltage into the memristors and propagate spike to the next layer
                for (auto& axon_terminal: axon_terminals) {
                    axon_terminal->receive_spike();
                    network->inject_spike(spike{timestamp, axon_terminal.get(), spike_type::generated});
                }
                
            // programming spike
            } else if (type == spike_type::programming) {
                // 1. increase in voltage square waveform (2.1V)
                potential += input_voltage;
                
                // 2. end of integration spike on the same neuron after tau_up
                network->inject_spike(spike{timestamp + tau_up, s, spike_type::end_of_integration});
                
                // 3. inject voltage into the memristors
                for (auto& axon_terminal: axon_terminals) {
                    axon_terminal->receive_spike();
                }
                
            // spike that resets the potential to get the square waveform
            } else if (type == spike_type::end_of_integration) {
                potential = resting_potential;
                
                // reset the synaptic current for all the memristors
                for (auto& axon_terminal: axon_terminals) {
                    axon_terminal->update(timestamp, 0);
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
                {"tau_up", tau_up},
                {"tau_down", tau_down},
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
        
        // ----- PULSE_GENERATOR PARAMETERS -----
        double    tau_up;
        double    tau_down;
        float     input_voltage;
    };
}
