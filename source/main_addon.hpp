/*
 * main_addon.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/07/2018
 *
 * Information: Expands the addon class, used for add-ons that need to be run on the main thread such as the Qt GUI
 */

#pragma once

#include "addon.hpp"

namespace hummus {
	class MainAddon : public Addon {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		MainAddon() = default;
		virtual ~MainAddon(){}
		
        // method that starts the GUI
		virtual void begin(Network* network, std::mutex* sync){}
        
        // method to reset the GUI
        virtual void reset() {}
	};
}
