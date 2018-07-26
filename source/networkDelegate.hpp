/*
 * networkDelegate.hpp
 * Adonis_t - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 17/01/2018
 *
 * Information: the networkDelegate class is polymorphic class to handle add-ons
 */

#pragma once

namespace adonis_t
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
		
		enum class Mode
		{
			display,
			spikeLogger,
			learningLogger
		};
		
		// ----- PURE VIRTUAL METHOD -----
		virtual void getArrivingSpike(double timestamp, projection* p, bool spiked, bool empty, Network* network, Neuron* postNeuron, const std::vector<double>& timeDifferences, const std::vector<std::vector<int16_t>>& plasticNeurons) = 0;
		virtual Mode getMode() const = 0;
	};
}
