/*
 * mainThreadAddon.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/07/2018
 *
 * Information: Expands the addon class, used for add-ons that need to be run on the main thread such as the Qt or Puffin GUI
 */

#pragma once

#include "addon.hpp"

namespace hummus {
	class MainThreadAddon : public Addon {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		MainThreadAddon() = default;
		virtual ~MainThreadAddon(){}
		
        // method that starts the GUI
		virtual void begin(Network* network, std::mutex* sync){}
        
        // method that is used to send an update to the GUI when an asynchronous network is used. This helps approximate the potential curve on the GUI.
        virtual void statusUpdate(double timestamp, synapse* a, Network* network){}
	};
}
