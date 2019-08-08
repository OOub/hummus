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
    // synapse models enum for readability
    enum class synapseType {
        excitatory,
        inhibitory
    };
    
    class Synapse {
    public:
        
        // ----- CONSTRUCTOR AND DESTRUCTOR -----
        Synapse(int _postsynaptic_neuron, int _presynaptic_neuron, float _weight, float _delay, synapseType _type, float _externalCurrent=100) :
                presynaptic_neuron(_presynaptic_neuron),
                postsynaptic_neuron(_postsynaptic_neuron),
                weight(_weight),
                delay(_delay),
                type(_type),
                externalCurrent(_externalCurrent),
                synapticCurrent(0),
                previousInputTime(0),
                gaussianStdDev(0),
                json_id(0),
                kernelID(0),
                synapticEfficacy(1),
                synapseTimeConstant(0) {}
        
        virtual ~Synapse(){}
        
        // ----- PUBLIC SYNAPSE METHODS -----
        
        // pure virtuak method that updates the current value in the absence of a spike
        virtual float update(double timestamp) = 0;
        
        // pure virtual method that outputs an updated current value upon receiving a spike
        virtual void receiveSpike(double timestamp) = 0;
        
        // write synapse parameters in a JSON format
        virtual void toJson(nlohmann::json& output) {}
        
        // resets the synapse
        virtual void reset() {
            synapticCurrent = 0;
        }
        
        // ----- SETTERS AND GETTERS -----
        synapseType getType() const {
            return type;
        }
        
        int getJsonId() const {
            return json_id;
        }
        
        float getSynapticCurrent() const {
            return synapticCurrent;
        }
        
        double getPreviousInputTime() const {
            return previousInputTime;
        }
        
        void setPreviousInputTime(double newTime) {
            previousInputTime = newTime;
        }
        
        const float getSynapseTimeConstant() const {
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
                if (weight > 0) {
                    weight += newWeight;
                    // prevent weights from being negative
                    if (weight < 0) {
                        weight = 0;
                    }
                }
            } else {
                weight = newWeight;
            }
        }
        
        float getDelay() const {
            return delay;
        }
        
        void setDelay(float newDelay, bool increment=true) {
            if (increment) {
                if (delay > 0) {
                    delay += newDelay;
                    // prevent delays from being negative
                    if (delay < 0) {
                        delay = 0;
                    }
                }
            } else {
                delay = newDelay;
                // prevent delays from being negative
                if (delay < 0) {
                    delay = 0;
                    std::cout << "negative delay set to 0" << std::endl;
                }
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
        float                      synapticCurrent;
        double                     previousInputTime;
        int                        kernelID;
        float                      gaussianStdDev;
        int                        json_id;
        float                      synapseTimeConstant;
        float                      externalCurrent;
        float                      synapticEfficacy;
        synapseType                type;
    };
}
