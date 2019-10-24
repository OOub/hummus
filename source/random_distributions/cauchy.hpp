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
        Cauchy(double weight_location=1, double weight_scale=0, double delay_location=0, double delay_scale=0) {

            // randomising weights and delays
            std::random_device device;
            random_engine = std::mt19937(device());
            delay_random = std::cauchy_distribution<double>(delay_location, delay_scale);
            weight_random = std::cauchy_distribution<double>(weight_location, weight_scale);
        }
		
        std::pair<double, double> operator()(int x, int y, int depth) {
			return std::make_pair(weight_random(random_engine), std::abs(delay_random(random_engine)));
        }
        
    protected :
        
        // ----- IMPLEMENTATION VARIABLES -----
        std::mt19937                     random_engine;
        std::cauchy_distribution<double> delay_random;
        std::cauchy_distribution<double> weight_random;
	};
}

