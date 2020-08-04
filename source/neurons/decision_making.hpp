/*
 * decision_making.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/01/2019
 *
 * Information: Decision-making neurons act as our classifier, roughly approximating a histogram activity-dependent classification. They should always be on the last layer of a network.
 */

#pragma once

namespace hummus {

    class Synapse;
    class Neuron;
    class Network;

	class Decision_Making : public Neuron {
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        Decision_Making(int _neuronID, int _layerID, int _sublayerID, int _rf_id,  std::pair<int, int> _xyCoordinates, int _classLabel=-1, float _threshold=-50, float _restingPotential=-70) :
                Neuron(_neuronID, _layerID, _sublayerID, _rf_id, _xyCoordinates, 0, 200, 10, 20, _threshold, _restingPotential, _classLabel) {
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
            
            if (type == spike_type::decision) {
                                    
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
                
            } else if (type == spike_type::generated) {
                ++intensity;
            }
        }

        // reset a neuron to its initial status
        virtual void reset_neuron(Network* network, bool clearAddons=true) override {
            intensity = 0;
            potential = resting_potential;
            previous_spike_time = 0;

            for (auto& dendrite: dendritic_tree) {
                dendrite->reset();
            }

            if (clearAddons) {
                relevant_addons.clear();
            }
        }

        virtual float share_information() override {
            return static_cast<float>(intensity);
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
