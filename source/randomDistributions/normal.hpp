/*
 * normal.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 28/02/2019
 *
 * Information: The Normal class can be used as an input for network methods that require lambda function to connect layers with weights and delays following a particular distribution (eg. allToAll). In this case, the distribution is a normal distribution. Delays are always positive so we take the absolute value of the random output
 */

#pragma once

#include <random>
#include <cmath>

namespace hummus {
	
	class Normal {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        Normal(float _weight_mu=1, float _weight_sigma=0, float _delay_mu=0, float _delay_sigma=0, float _weight_lower_limit=-INFINITY, float _weight_upper_limit=INFINITY, float _delay_lower_limit=0, float _delay_upper_limit=INFINITY) :
                weight_mu(_weight_mu),
                weight_sigma(_weight_sigma),
                weight_lower_limit(_weight_lower_limit),
                weight_upper_limit(_weight_upper_limit),
                delay_mu(_delay_mu),
                delay_sigma(_delay_sigma),
                delay_lower_limit(_delay_lower_limit),
                delay_upper_limit(_delay_upper_limit) {
            // randomising weights and delays
            std::random_device device;
            random_engine = std::mt19937(device());
            delay_random = std::normal_distribution<float>(delay_mu, delay_sigma);
            weight_random = std::normal_distribution<float>(weight_mu, weight_sigma);
        }
		
        std::pair<float, float> operator()(int x, int y, int depth) {
            return std::make_pair(truncate(weight_random(random_engine), weight_lower_limit, weight_upper_limit), truncate(delay_random(random_engine), delay_lower_limit, delay_upper_limit));
        }
		
        // truncated normal distribution
        float truncate(float x, float a, float b) {
            if (x >= a && x <= b) {
                return x;
            } else {
                return 0;
            }
        }
		
		
    protected :
        
        // ----- IMPLEMENTATION VARIABLES -----
        std::mt19937                    random_engine;
        std::normal_distribution<float> delay_random;
        std::normal_distribution<float> weight_random;
        float                           weight_mu;
        float                           weight_sigma;
        float                           weight_lower_limit;
        float                           weight_upper_limit;
        float                           delay_mu;
        float                           delay_sigma;
        float                           delay_lower_limit;
        float                           delay_upper_limit;
	};
}

