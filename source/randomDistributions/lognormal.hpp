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
        LogNormal(float weightMean=1, float weightStdDev=0, float delayMean=0, float delayStdDev=0) {

            // randomising weights and delays
            std::random_device device;
            randomEngine = std::mt19937(device());
            delayRandom = std::lognormal_distribution<>(delayMean, delayStdDev);
            weightRandom = std::lognormal_distribution<>(weightMean, weightStdDev);
        }
		
        std::pair<float, float> operator()(int16_t x, int16_t y, int16_t depth) {
			return std::make_pair(weightRandom(randomEngine), delayRandom(randomEngine));
        }
        
    protected :
        
        // ----- IMPLEMENTATION VARIABLES -----
        std::mt19937                  randomEngine;
        std::lognormal_distribution<> delayRandom;
        std::lognormal_distribution<> weightRandom;
        bool                          weightSameSign;
	};
}

