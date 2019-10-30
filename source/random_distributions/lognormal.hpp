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
        LogNormal(float weight_mu=1, float weight_sigma=0, float delay_mu=0, float delay_sigma=0) {

            // randomising weights and delays
            std::random_device device;
            random_engine = std::mt19937(device());
            delay_random = std::lognormal_distribution<float>(delay_mu, delay_sigma);
            weight_random = std::lognormal_distribution<float>(weight_mu, weight_sigma);
        }
		
        std::pair<float, float> operator()(int x, int y, int depth) {
			return std::make_pair(weight_random(random_engine), delay_random(random_engine));
        }
        
    protected :
        
        // ----- IMPLEMENTATION VARIABLES -----
        std::mt19937                        random_engine;
        std::lognormal_distribution<float> delay_random;
        std::lognormal_distribution<float> weight_random;
	};
}

