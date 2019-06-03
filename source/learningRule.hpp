/*
 * learningRule.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/05/2019
 *
 * Information: The learning class is polymorphic class to handle learning rules
 */

#pragma once

namespace hummus {
    
    class Network;
    
    class LearningRule {
    public:
        
        // ----- CONSTRUCTOR AND DESTRUCTOR -----
        LearningRule() = default;
        virtual ~LearningRule(){}
        
        // ----- PUBLIC METHODS -----
        
        // message that is activated before the network starts running
        virtual void initialise(Network* network){}
        
        // message that is activated whenever a synapse wants to learn
        virtual void learn(double timestamp, Network* network){};
    };
}
