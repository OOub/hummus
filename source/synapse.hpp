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

#include "dependencies/json.hpp"

namespace hummus {
    class Synapse {
    public:
        
        // ----- CONSTRUCTOR AND DESTRUCTOR -----
        Synapse(size_t _postsynaptic_neuron, size_t _presynaptic_neuron, float _weight, float _delay, float _externalCurrent=100) :
                presynaptic_neuron(_presynaptic_neuron),
                postsynaptic_neuron(_postsynaptic_neuron),
                weight(_weight),
                delay(_delay),
                externalCurrent(_externalCurrent),
                previousInputTime(0),
                gaussianStdDev(0),
                type(0),
                kernelID(0),
                synapseTimeConstant(0) {}
        
        virtual ~Synapse(){}
        
        // ----- PUBLIC SYNAPSE METHODS -----
        
        // pure virtual method that updates the status of current before integrating a spike
        virtual double update(double timestamp, double previousTime, float neuronCurrent) = 0;
        
        // pure virtual method that outputs an updated current value
        virtual float receiveSpike(float neuronCurrent) = 0;
        
        // write synapse parameters in a JSON format
        virtual void toJson(nlohmann::json& output) {}
        
        // ----- SETTERS AND GETTERS -----
        double getPreviousInputTime() const {
            return previousInputTime;
        }
        
        void setPreviousInputTime(double newTime) {
            previousInputTime = newTime;
        }
        
        float getSynapseTimeConstant() const {
            return synapseTimeConstant;
        }
        
        size_t getPresynapticNeuronID() const {
            return presynaptic_neuron;
        }
        
        size_t getPostsynapticNeuronID() const {
            return postsynaptic_neuron;
        }
        
        float getWeight() const {
            return weight;
        }
        
        void setWeight(float newWeight, bool increment=true) {
            if (increment) {
                weight += newWeight;
            } else {
                weight = newWeight;
            }
        }
        
        float getDelay() const {
            return delay;
        }
        
        void setDelay(float newDelay, bool increment=true) {
            if (increment) {
                delay += newDelay;
            } else {
                delay = newDelay;
            }
        }
        
    protected:
        size_t                        presynaptic_neuron;
        size_t                        postsynaptic_neuron;
        float                         weight;
        float                         delay;
        float                         externalCurrent;
        double                        previousInputTime;
        int                           kernelID;
        float                         gaussianStdDev;
        int                           type;
        float                         synapseTimeConstant;
    };
}
