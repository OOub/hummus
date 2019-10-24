/*
 * synapse.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/05/2019
 *
 * Information: Implementation of a synapse. Each neuron contains a collection of synapses
 */

#pragma once

#include "../third_party/json.hpp"

namespace hummus {
    // synapse models enum for readability
    enum class synapse_type {
        excitatory,
        inhibitory
    };

    class Synapse {
    public:

        // ----- CONSTRUCTOR AND DESTRUCTOR -----
        Synapse(int _postsynaptic_neuron, int _presynaptic_neuron, double _weight, double _delay, double _external_current=100) :
                presynaptic_neuron(_presynaptic_neuron),
                postsynaptic_neuron(_postsynaptic_neuron),
                weight(_weight),
                delay(_delay),
                synaptic_current(0),
                synaptic_potential(0),
                previous_input_time(0),
                gaussian_std_dev(0),
                synapse_time_constant(0),
                external_current(_external_current),
                synaptic_efficacy(1),
                kernel_id(0),
                json_id(0) {}

        virtual ~Synapse(){}

        // ----- PUBLIC SYNAPSE METHODS -----

        // pure virtual method that updates the current value in the absence of a spike
        virtual double update(double timestamp, double timestep=0) { return 0; };

        // pure virtual method that updates the synaptic current upon receiving a spike
        virtual void receive_spike(double potential=0) {};

        // write synapse parameters in a JSON format
        virtual void to_json(nlohmann::json& output) {
            output.push_back({
                {"json_id", json_id},
                {"weight", weight},
                {"delay", delay},
                {"postsynaptic_neuron", postsynaptic_neuron},
            });
        }

        // resets the synapse
        virtual void reset() {
            previous_input_time = 0;
            synaptic_current = 0;
            synaptic_potential = 0;
        }

        // ----- SETTERS AND GETTERS -----
        synapse_type get_type() const {
            return type;
        }

        int get_json_id() const {
            return json_id;
        }

        double get_synaptic_potential() const {
            return synaptic_potential;
        }
        
        double get_synaptic_current() const {
            return synaptic_current;
        }

        double get_previous_input_time() const {
            return previous_input_time;
        }

        void set_previous_input_time(double new_time) {
            previous_input_time = new_time;
        }

        double get_synapse_time_constant() const {
            return synapse_time_constant;
        }

        int get_presynaptic_neuron_id() const {
            return presynaptic_neuron;
        }

        int get_postsynaptic_neuron_id() const {
            return postsynaptic_neuron;
        }

        double get_weight() const {
            return weight;
        }

        void set_weight(double new_weight) {
            weight = new_weight;
        }

        void increment_weight(double delta_weight) {
            if (weight > 0) {
                weight += delta_weight;
                // prevent weights from being negative
                if (weight < 0) {
                    weight = 0;
                }
            }
        }

        double get_delay() const {
            return delay;
        }

        void set_delay(double new_delay) {
            delay = new_delay;
        }

        void increment_delay(double delta_delay) {
            if (delay > 0) {
                delay += delta_delay;
                // prevent delays from being negative
                if (delay < 0) {
                    delay = 0;
                }
            }
        }

        double get_synaptic_efficacy() const {
            return synaptic_efficacy;
        }

        void set_synaptic_efficacy(double new_efficacy) {
            synaptic_efficacy = new_efficacy;
        }

        void increment_synaptic_efficacy(double delta_efficacy) {
            synaptic_efficacy += delta_efficacy;
        }

    protected:
        int                        presynaptic_neuron;
        int                        postsynaptic_neuron;
        double                     weight;
        double                     delay;
        double                     synaptic_current;
        double                     synaptic_potential;
        double                     previous_input_time;
        double                     gaussian_std_dev;
        double                     synapse_time_constant;
        double                     external_current;
        double                     synaptic_efficacy;
        synapse_type               type;
        int                        kernel_id;
        int                        json_id;
    };
}
