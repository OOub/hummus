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
        Synapse(int _postsynaptic_neuron, int _presynaptic_neuron, float _weight, float _delay, float _externalCurrent=100) :
                presynaptic_neuron(_presynaptic_neuron),
                postsynaptic_neuron(_postsynaptic_neuron),
                weight(_weight),
                delay(_delay),
                externalCurrent(_externalCurrent),
                synapticCurrent(0),
                previousInputTime(0),
                gaussianStdDev(0),
                type(0),
                kernelID(0),
                synapticEfficacy(1),
                synapseTimeConstant(0) {}
        
        virtual ~Synapse(){}
        
        // ----- PUBLIC SYNAPSE METHODS -----
        
        // pure virtual method that outputs an updated current value
        virtual float receiveSpike(double timestamp) = 0;
        
        // write synapse parameters in a JSON format
        virtual void toJson(nlohmann::json& output) {}
        
        // ----- SETTERS AND GETTERS -----
        int getType() const {
            return type;
        }
        
        float getSynapticCurrent() const {
            return synapticCurrent;
        }
        
        double getPreviousInputTime() const {
            return previousInputTime;
        }

        float getSynapseTimeConstant() const {
            return synapseTimeConstant;
        }
        
        int getPresynapticNeuronID() const {
            return presynaptic_neuron;
        }
        
        int getPostsynapticNeuronID() const {
            return postsynaptic_neuron;
        }
        
        float getWeight() const {
            return weight;
        }
        
        void setWeight(float newWeight, bool increment=true) {
            if (increment) {
                weight += synapticEfficacy * newWeight;
            } else {
                weight = newWeight;
            }
        }
        
        float getDelay() const {
            return delay;
        }
        
        void setDelay(float newDelay, bool increment=true) {
            if (increment) {
                delay += synapticEfficacy * newDelay;
            } else {
                delay = newDelay;
            }
        }
        
        float getSynapticEfficacy() const {
            return synapticEfficacy;
        }
        
        void setSynapticEfficacy(float newEfficacy, bool increment=true) {
            if (increment) {
                synapticEfficacy += newEfficacy;
            } else{
                synapticEfficacy = newEfficacy;
            }
        }
        
    protected:
        int                        presynaptic_neuron;
        int                        postsynaptic_neuron;
        float                      weight;
        float                      delay;
        float                      externalCurrent;
        float                      synapticCurrent;
        double                     previousInputTime;
        int                        kernelID;
        float                      gaussianStdDev;
        int                        type;
        float                      synapseTimeConstant;
        float                      synapticEfficacy;
    };
}
