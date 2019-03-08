/*
 * normal.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 28/02/2019
 *
 * Information: The Normal class can be used as an input for network methods that require lambda function to connect layers with weights and delays following a particular distribution (eg. allToAll). In this case, the distribution is a normal distribution
 */

#pragma once

#include <random>
#include <cmath>

namespace hummus {
	
	class Normal {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        Normal(float weightMean=1, float weightStdDev=0, float delayMean=0, float delayStdDev=0, bool _weightSameSign=true) :
        		weightSameSign(_weightSameSign) {

            // randomising weights and delays
            std::random_device device;
            randomEngine = std::mt19937(device());
            delayRandom = std::normal_distribution<>(delayMean, delayStdDev);
            weightRandom = std::normal_distribution<>(weightMean, weightStdDev);

            // all weights positive if mean weight is positive and vice-versa
            sign = weightMean<0?-1:weightMean>=0;
        }
		
        std::pair<float, float> operator()(int16_t x, int16_t y, int16_t depth) {
        	if (weightSameSign) {
            	return std::make_pair(sign*std::abs(weightRandom(randomEngine)), std::abs(delayRandom(randomEngine)));
			} else {
				return std::make_pair(weightRandom(randomEngine), std::abs(delayRandom(randomEngine)));
			}
        }
        
    protected :
        
        // ----- IMPLEMENTATION VARIABLES -----
        int                        sign;
        std::mt19937               randomEngine;
        std::normal_distribution<> delayRandom;
        std::normal_distribution<> weightRandom;
        bool                       weightSameSign;
	};
}

