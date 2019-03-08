/*
 * cauchy.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 28/02/2019
 *
 * Information: The Cauchy class can be used as an input for network methods that require lambda function to connect layers with weights and delays following a particular distribution (eg. allToAll). In this case, the distribution is a cauchy distribution. Delays are always positive so we take the absolute value of the random output
 */

#pragma once

#include <random>
#include <cmath>

namespace hummus {
	
	class Cauchy {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        Cauchy(float weightLocation=1, float weightScale=0, float delayLocation=0, float delayScale=0) {

            // randomising weights and delays
            std::random_device device;
            randomEngine = std::mt19937(device());
            delayRandom = std::cauchy_distribution<>(delayLocation, delayScale);
            weightRandom = std::cauchy_distribution<>(weightLocation, weightScale);
        }
		
        std::pair<float, float> operator()(int16_t x, int16_t y, int16_t depth) {
			return std::make_pair(weightRandom(randomEngine), std::abs(delayRandom(randomEngine)));
        }
        
    protected :
        
        // ----- IMPLEMENTATION VARIABLES -----
        std::mt19937               randomEngine;
        std::cauchy_distribution<> delayRandom;
        std::cauchy_distribution<> weightRandom;
	};
}

