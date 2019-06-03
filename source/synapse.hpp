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

#include "learningRule.hpp"
#include "dependencies/json.hpp"

namespace hummus {
    class Synapse {
    public:
        
        // ----- CONSTRUCTOR AND DESTRUCTOR -----
        Synapse(Synapse* _target_neuron, Synapse* _parent_neuron, float _weight, float _delay) :
                parent_neuron(_parent_neuron),
                target_neuron(_target_neuron),
                weight(_weight),
                delay(_delay),
                previousInputTime(0),
                gaussianStdDev(0),
                type(0),
                kernelID(0),
                synapseTimeConstant(0) {}
        
        virtual ~Synapse(){}
        
        // ----- PUBLIC SYNAPSE METHODS -----
        
        // initialises a learning rule
        template <typename T, typename... Args>
        T& makeLearningRule(Args&&... args) {
            rule.reset(new T(std::forward<Args>(args)...));
            return static_cast<T&>(*rule);
        }
        
        // pure virtual method that updates the status of current before integrating a spike
        virtual double update(double timestamp, double timestep, float neuronCurrent) = 0;
        
        // pure virtual method that outputs an updated current value
        virtual float receiveSpike(float neuronCurrent, float externalCurrent, float synapseWeight) = 0;
        
        // write synapse parameters in a JSON format
        virtual void toJson(nlohmann::json& output) {}
        
        // ----- SETTERS AND GETTERS -----
        float getSynapseTimeConstant() const {
            return synapseTimeConstant;
        }
        
    protected:
        Synapse*                      parent_neuron;
        Synapse*                      target_neuron;
        float                         weight;
        float                         delay;
        double                        previousInputTime;
        std::unique_ptr<LearningRule> rule;
        int                           kernelID;
        float                         gaussianStdDev;
        int                           type;
        float                         synapseTimeConstant;
    };
}
