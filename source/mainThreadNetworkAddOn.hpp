/*
 * mainThreadNetworkAddOn.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/07/2018
 *
 * Information: Expands on the NetworkDelegate class, used for add-ons that need to be run on the main thread.
 */

#pragma once

#include "networkAddOn.hpp"

namespace adonis_c
{
	class MainThreadNetworkAddOn : public NetworkAddOn
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		MainThreadNetworkAddOn() = default;
		virtual ~MainThreadNetworkAddOn(){}
		
		virtual void begin(Network* network, std::mutex* sync){}
	};
}
