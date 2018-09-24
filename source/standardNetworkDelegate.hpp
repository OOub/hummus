/*
 * standardNetworkDelegate.hpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 12/06/2018
 *
 * Information: Expands on the NetworkDelegate class, used for add-ons that don't specifically need to be run on the main thread
 */

#pragma once

#include "networkDelegate.hpp"

namespace adonis_c
{
	class StandardNetworkDelegate : public NetworkDelegate
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		StandardNetworkDelegate(){}
		virtual ~StandardNetworkDelegate(){}
	};
}
