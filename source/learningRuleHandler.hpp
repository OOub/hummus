/*
 * learningRuleHandler.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 25/09/2018
 *
 * Information: The LearningRuleHandler class is called from a neuron to apply local learning rules.
 */

#pragma once

namespace adonis {
	class Network;
	class Neuron;
	
	class LearningRuleHandler {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		LearningRuleHandler() = default;
		
		// ----- PUBLIC METHODS -----
        
        // pure virtual function that needs to be implemented in every learning rule. The body would contain said learning rule. This specific method should be called inside a neuron's requestLearn method
		virtual void learn(double timestamp, Neuron* neuron, Network* network) = 0;
	};
}

