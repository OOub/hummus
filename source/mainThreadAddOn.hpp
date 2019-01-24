/*
 * mainThreadAddOn.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/07/2018
 *
 * Information: Expands on the addOn class, used for add-ons that need to be run on the main thread such as the Qt GUI
 */

#pragma once

#include "addOn.hpp"

namespace adonis
{
	class MainThreadAddOn : public AddOn
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		MainThreadAddOn() = default;
		virtual ~MainThreadAddOn(){}
		
		virtual void begin(Network* network, std::mutex* sync){}
	};
}
