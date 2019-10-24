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
        LogNormal(double weight_mu=1, double weight_sigma=0, double delay_mu=0, double delay_sigma=0) {

            // randomising weights and delays
            std::random_device device;
            random_engine = std::mt19937(device());
            delay_random = std::lognormal_distribution<double>(delay_mu, delay_sigma);
            weight_random = std::lognormal_distribution<double>(weight_mu, weight_sigma);
        }
		
        std::pair<double, double> operator()(int x, int y, int depth) {
			return std::make_pair(weight_random(random_engine), delay_random(random_engine));
        }
        
    protected :
        
        // ----- IMPLEMENTATION VARIABLES -----
        std::mt19937                        random_engine;
        std::lognormal_distribution<double> delay_random;
        std::lognormal_distribution<double> weight_random;
	};
}

