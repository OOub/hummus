/*
 * computerVisionExtention.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 28/02/2019
 *
 * Information: The ComputerVisionExtention class extends the Network class to facilitate tasks related to computer vision
 */

#pragma once

#include "../core.hpp"

namespace hummus {
	class ComputerVisionExtention : public Network {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        ComputerVisionExtention() = default;
        
		virtual ~ComputerVisionExtention(){}
        
	};
}
