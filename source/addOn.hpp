/*
 * addOn.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 24/01/2019
 *
 * Information: The addOn class is polymorphic class to handle add-ons. It contains a series of methods acting as messages that can be used throughout the network for different purposes
 */

#pragma once

namespace adonis {
    class Neuron;
    class Network;
	struct axon;
	
	// polymorphic class for add-ons
	class AddOn {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		AddOn() = default;
		virtual ~AddOn(){}
		
		// ----- PUBLIC METHODS -----
        
        // message that is actived before the network starts running
		virtual void onStart(Network* network){}
        
        // message that is actived when the network finishes running
		virtual void onCompleted(Network* network){}
        
        // message that is activated whenever a neuron receives a spike
		virtual void incomingSpike(double timestamp, axon* a, Network* network){}
        
        // message that is activated whenever a neuron emits a spike
		virtual void neuronFired(double timestamp, axon* a, Network* network){}
        
        // message that is activated on every timestep on the synchronous network only. This allows decay equations and the GUI to keep calculating even when neurons don't receive any spikes
		virtual void timestep(double timestamp, Network* network, Neuron* postNeuron){}
	};
}
