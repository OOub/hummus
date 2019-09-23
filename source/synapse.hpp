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
        Synapse(int _postsynaptic_neuron, int _presynaptic_neuron, float _weight, float _delay, float _external_current=100) :
                presynaptic_neuron(_presynaptic_neuron),
                postsynaptic_neuron(_postsynaptic_neuron),
                weight(_weight),
                delay(_delay),
                external_current(_external_current),
                synaptic_current(0),
                previous_input_time(0),
                gaussian_std_dev(0),
                json_id(0),
                kernel_id(0),
                synaptic_efficacy(1),
                synapse_time_constant(0) {}
        
        virtual ~Synapse(){}
        
        // ----- PUBLIC SYNAPSE METHODS -----
        
        // pure virtual method that updates the current value in the absence of a spike
        virtual float update(double timestamp, float timestep, bool asynchronous) = 0;
        
        // pure virtual method that outputs an updated current value upon receiving a spike
        virtual void receive_spike() = 0;
        
        // write synapse parameters in a JSON format
        virtual void to_json(nlohmann::json& output) {}
        
        // resets the synapse
        virtual void reset() {
            synaptic_current = 0;
        }
        
        // ----- SETTERS AND GETTERS -----
        synapse_type get_type() const {
            return type;
        }
        
        int get_json_id() const {
            return json_id;
        }
        
        float get_synaptic_current() const {
            return synaptic_current;
        }
        
        double get_previous_input_time() const {
            return previous_input_time;
        }
        
        void set_previous_input_time(double new_time) {
            previous_input_time = new_time;
        }
        
        const float get_synapse_time_constant() const {
            return synapse_time_constant;
        }
        
        int get_presynaptic_neuron_id() const {
            return presynaptic_neuron;
        }
        
        int get_postsynaptic_neuron_id() const {
            return postsynaptic_neuron;
        }
        
        float get_weight() const {
            return weight;
        }
        
        void set_weight(float new_weight) {
            weight = new_weight;
        }
        
        void increment_weight(float delta_weight) {
            if (weight > 0) {
                weight += delta_weight;
                // prevent weights from being negative
                if (weight < 0) {
                    weight = 0;
                }
            }
        }
        
        float get_delay() const {
            return delay;
        }
        
        void set_delay(float new_delay) {
            delay = new_delay;
        }
        
        void increment_delay(float delta_delay) {
            if (delay > 0) {
                delay += delta_delay;
                // prevent delays from being negative
                if (delay < 0) {
                    delay = 0;
                }
            }
        }
        
        float get_synaptic_efficacy() const {
            return synaptic_efficacy;
        }
        
        void set_synaptic_efficacy(float new_efficacy) {
            synaptic_efficacy = new_efficacy;
        }
        
        void increment_synaptic_efficacy(float delta_efficacy) {
            synaptic_efficacy += delta_efficacy;
        }
        
    protected:
        int                        presynaptic_neuron;
        int                        postsynaptic_neuron;
        float                      weight;
        float                      delay;
        float                      synaptic_current;
        double                     previous_input_time;
        float                      gaussian_std_dev;
        float                      synapse_time_constant;
        float                      external_current;
        float                      synaptic_efficacy;
        synapse_type               type;
        int                        kernel_id;
        int                        json_id;
    };
}
