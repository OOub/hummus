/*
 * standardAddOn.hpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 12/06/2018
 *
 * Information: Expands on the addOn class, used for add-ons that don't specifically need to be run on the main thread
 */

#pragma once

#include "addOn.hpp"

namespace adonis
{
	class StandardAddOn : public AddOn
	{
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		StandardAddOn() = default;
		virtual ~StandardAddOn(){}
	};
}
