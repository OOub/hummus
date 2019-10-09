/*
 * pulse_generator.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/01/2019
 *
 * Information: Parrot neurons take in spikes or events and instantly propagate them in the network. The potential does not decay.
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
        Pulse_Generator(int _neuronID, int _layerID, int _sublayerID, int _rf_id,  std::pair<int, int> _xyCoordinates) :
                Neuron(_neuronID, _layerID, _sublayerID, _rf_id, _xyCoordinates, 0, 0, 0, 0, 0, 0, "") {

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
            // initial or generated -> inference pulse
            // 1. decrease in voltage square waveform (1.1V)
            // 2. end of integration spike on the same neuron after tau_up

            // programming -> programming pulse
            // 1. increase in voltage square waveform (1.1V)
            // 2. end of integration spike on the same neuron after tau_down
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
	};
}
