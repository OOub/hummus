/*
 * mainThreadNetworkDelegate.hpp
 * Adonis_t - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/07/2018
 *
 * Information: Expands on the NetworkDelegate class, used for add-ons that need to be run on the main thread.
 */

#pragma once

#include "networkDelegate.hpp"

namespace adonis_t
{
	class MainThreadNetworkDelegate : public NetworkDelegate
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		MainThreadNetworkDelegate() = default;
		virtual ~MainThreadNetworkDelegate(){}
		
		virtual void begin(int numberOfLayers, std::vector<int> neuronsInLayers){}
	};
}
