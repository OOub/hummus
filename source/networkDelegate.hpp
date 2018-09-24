/*
 * networkDelegate.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 17/01/2018
 *
 * Information: The NetworkDelegate class is polymorphic class to handle add-ons
 */

#pragma once

namespace adonis_c
{
	class Network;
	class Neuron;
	struct projection;
	
	// polymorphic class for add-ons
	class NetworkDelegate
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		NetworkDelegate() = default;
		virtual ~NetworkDelegate(){}
		
		// ----- PUBLIC METHODS -----
		virtual void incomingSpike(double timestamp, projection* p, Network* network){}
		virtual void neuronFired(double timestamp, projection* p, Network* network){}
		virtual void timestep(double timestamp, Network* network, Neuron* postNeuron){}
		virtual void learningEpoch(double timestamp, Network* network, Neuron* postNeuron, const std::vector<double>& timeDifferences, const std::vector<std::vector<int16_t>>& plasticNeurons){}
	};
}
