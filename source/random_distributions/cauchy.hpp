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
        Cauchy(float weight_location=1, float weight_scale=0, float delay_location=0, float delay_scale=0) {

            // randomising weights and delays
            delay_random = std::cauchy_distribution<float>(delay_location, delay_scale);
            weight_random = std::cauchy_distribution<float>(weight_location, weight_scale);
        }
		
        template<class RNG>
        std::pair<float, float> operator()(int x, int y, int depth, RNG &random_engine) {
			return std::make_pair(weight_random(random_engine), std::abs(delay_random(random_engine)));
        }
        
    protected :
        
        // ----- IMPLEMENTATION VARIABLES -----
        std::cauchy_distribution<float> delay_random;
        std::cauchy_distribution<float> weight_random;
	};
}

