/*
 * networkAddOn.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 17/01/2018
 *
 * Information: The NetworkAddOn class is polymorphic class to handle add-ons
 */

#pragma once

namespace adonis_c
{
	class Network;
	class Neuron;
	struct axon;
	
	// polymorphic class for add-ons
	class NetworkAddOn
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		NetworkAddOn() = default;
		virtual ~NetworkAddOn(){}
		
		// ----- PUBLIC METHODS -----
		virtual void onCompleted(Network* network){}
		virtual void incomingSpike(double timestamp, axon* a, Network* network){}
		virtual void neuronFired(double timestamp, axon* a, Network* network){}
		virtual void timestep(double timestamp, Network* network, Neuron* postNeuron){}
		virtual void learningEpoch(double timestamp, Network* network, Neuron* postNeuron, const std::vector<double>& timeDifferences, const std::vector<std::vector<int16_t>>& plasticNeurons){}
	};
}
