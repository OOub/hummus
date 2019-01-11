/*
 * addOn.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 17/01/2018
 *
 * Information: The addOn class is polymorphic class to handle add-ons
 */

#pragma once

namespace adonis
{
    class Neuron;
    class Network;
	struct axon;
	
	// polymorphic class for add-ons
	class AddOn
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		AddOn() = default;
		virtual ~AddOn(){}
		
		// ----- PUBLIC METHODS -----
		virtual void onStart(Network* network){}
		virtual void onCompleted(Network* network){}
		virtual void incomingSpike(double timestamp, axon* a, Network* network){}
		virtual void neuronFired(double timestamp, axon* a, Network* network){}
		virtual void timestep(double timestamp, Network* network, Neuron* postNeuron){}
	};
}
