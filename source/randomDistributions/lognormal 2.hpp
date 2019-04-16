/*
 * lognormal.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 28/02/2019
 *
 * Information: The LogNormal class can be used as an input for network methods that require lambda function to connect layers with weights and delays following a particular distribution (eg. allToAll). In this case, the distribution is a lognormal distribution
 */

#pragma once

#include <random>
#include <cmath>

namespace hummus {
	
	class LogNormal {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        LogNormal(float weightMu=1, float weightSigma=0, float delayMu=0, float delaySigma=0) {

            // randomising weights and delays
            std::random_device device;
            randomEngine = std::mt19937(device());
            delayRandom = std::lognormal_distribution<>(delayMu, delaySigma);
            weightRandom = std::lognormal_distribution<>(weightMu, weightSigma);
        }
		
        std::pair<float, float> operator()(int16_t x, int16_t y, int16_t depth) {
			return std::make_pair(weightRandom(randomEngine), delayRandom(randomEngine));
        }
        
    protected :
        
        // ----- IMPLEMENTATION VARIABLES -----
        std::mt19937                  randomEngine;
        std::lognormal_distribution<> delayRandom;
        std::lognormal_distribution<> weightRandom;
	};
}

