/*
 * networkDelegate.hpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 17/01/2018
 *
 * Information: the networkDelegate class is polymorphic class to handle add-ons
 */

#pragma once

namespace baal
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
			logger
		};
		
		// ----- PURE VIRTUAL METHOD -----
		virtual void getArrivingSpike(double timestamp, projection* p, bool spiked, bool empty, Network* network, Neuron* postNeuron) = 0;
		virtual Mode getMode() const = 0;
	};
}
