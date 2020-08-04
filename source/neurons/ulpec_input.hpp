/*
 * ulpec_input.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/01/2019
 *
 * Information: ULPEC_Input neurons generates waveforms used in the context of the ulpec project.
 */

#pragma once

namespace hummus {

    class Synapse;
    class Neuron;
    class Network;

    class ULPEC_Input : public Neuron {

    public:
        // ----- CONSTRUCTOR AND DESTRUCTOR -----
        ULPEC_Input(int _neuronID, int _layerID, int _sublayerID, int _rf_id,  std::pair<int, int> _xyCoordinates, int _refractoryPeriod=25, float _threshold=1.2, float _restingPotential=1.1, float _tau=10, float _injected_potential=-1) :
                Neuron(_neuronID, _layerID, _sublayerID, _rf_id, _xyCoordinates, _refractoryPeriod, 0, 0, 0, _threshold, _restingPotential, -1),
                injected_potential(_injected_potential) {

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

        virtual void update(double timestamp, Synapse* s, Network* network, float timestep, spike_type type) override {
            
            // updating GUI values status before any computation
            if (network->get_main_thread_addon()) {
                network->get_main_thread_addon()->status_update(timestamp, this, network);
            }
                
            // checking if there's a refractory period
            if (!active && timestamp - previous_spike_time >= refractory_period) {
                active = true;
            }

            if (type == spike_type::initial && active) {
                // updating potential of input neuron
                potential += injected_potential;
                
                // propagating spike to the next layer
                for (auto& axon_terminal: axon_terminals) {
                    network->inject_spike(spike{timestamp, axon_terminal.get(), spike_type::initial});
                }
                
                if (network->get_verbose() == 2) {
                    std::cout << "t " << timestamp << " " << neuron_id << " v_pre " << potential << " --> INPUT" << std::endl;
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
            }
        }

    protected:
        
        // ----- PULSE_GENERATOR PARAMETERS -----
        float       injected_potential;
    };
}
