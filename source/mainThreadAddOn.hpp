/*
 * mainThreadAddOn.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/07/2018
 *
 * Information: Expands on the addOn class, used for add-ons that need to be run on the main thread such as the Qt GUI
 */

#pragma once

#include "addOn.hpp"

namespace hummus {
	class MainThreadAddOn : public AddOn {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
		MainThreadAddOn() = default;
		virtual ~MainThreadAddOn(){}
		
        // method that starts the GUI
		virtual void begin(Network* network, std::mutex* sync){}
        
        // method that is used to send an update on a neuron's potential (before and after it changes) to the GUI when an asynchronous network is used.This helps approximate the potential curve on the GUI.
        virtual void statusUpdate(double timestamp, synapse* a, Network* network){}
	};
}
