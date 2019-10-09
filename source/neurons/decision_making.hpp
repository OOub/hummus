/*
 * decision_making.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/01/2019
 *
 * Information: Decision-making neurons act as our classifier, roughly approximating a histogram activity-dependent classification. They should always be on the last layer of a network.
 *
 * NEURON TYPE 2 (in JSON SAVE FILE)
 */

#pragma once

#include "../../third_party/json.hpp"

namespace hummus {

    class Synapse;
    class Neuron;
    class Network;

	class Decision_Making : public Neuron {
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        Decision_Making(int _neuronID, int _layerID, int _sublayerID, int _rf_id,  std::pair<int, int> _xyCoordinates, std::string _classLabel="", int _refractoryPeriod=0, float _traceTimeConstant=20, float _threshold=-50, float _restingPotential=-70) :
                Neuron(_neuronID, _layerID, _sublayerID, _rf_id, _xyCoordinates, _refractoryPeriod, 200, 10, _traceTimeConstant, _threshold, _restingPotential, _classLabel) {

            // DecisionMaking neuron type = 2 for JSON save
            neuron_type = 2;
        }

		virtual ~Decision_Making(){}

        // ----- PUBLIC DECISION MAKING NEURON METHODS -----
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

            // checking if the neuron is in a refractory period
            if (timestamp - previous_spike_time >= refractory_period) {
                active = true;
            }

            if (type == spike_type::decision) {
                // checking if the neuron is in a refractory period
                if (timestamp - previous_spike_time >= refractory_period) {
                    active = true;
                }

                potential = threshold;

                if (network->get_verbose() >= 1) {
                    std::cout << "t=" << timestamp << " class " << class_label << " --> DECISION" << std::endl;
                }

                for (auto& addon: relevant_addons) {
                    addon->neuron_fired(timestamp, s, this, network);
                }

                if (network->get_main_thread_addon()) {
                    network->get_main_thread_addon()->neuron_fired(timestamp, s, this, network);
                }

                // reset intensities on all other neurons
                winner_takes_all(timestamp, network);
                potential = resting_potential;
                previous_spike_time = timestamp;
                active = false;

            } else {
                if (type != spike_type::none && s->get_type() == synapse_type::excitatory){
                    if (active) {
                        ++intensity;
                    }
                }
            }
        }

        // reset a neuron to its initial status
        virtual void reset_neuron(Network* network, bool clearAddons=true) override {
            intensity = 0;
            potential = resting_potential;

            for (auto& dendrite: dendritic_tree) {
                dendrite->reset();
            }

            if (clearAddons) {
                relevant_addons.clear();
            }
        }

        virtual int share_information() override {
            return intensity;
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

        void winner_takes_all(double timestamp, Network* network) override {
            for (auto& n: network->get_layers()[layer_id].neurons) {
                auto& neuron = network->get_neurons()[n];
                dynamic_cast<Decision_Making*>(neuron.get())->set_intensity(0);
            }
        }

        // ----- SETTERS AND GETTERS -----
        void set_intensity(int new_intensity) {
            intensity = new_intensity;
        }

		// ----- DECISION-MAKING NEURON PARAMETERS -----
        int      intensity;
	};
}
