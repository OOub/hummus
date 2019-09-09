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
            delay_random = std::normal_distribution<>(delay_mu, delay_sigma);
            weight_random = std::normal_distribution<>(weight_mu, weight_sigma);
        }
		
        std::pair<float, float> operator()(int16_t x, int16_t y, int16_t depth) {
            return std::make_pair(truncate(weight_random(random_engine), weight_lower_limit, weight_upper_limit), truncate(delay_random(random_engine), delay_lower_limit, delay_upper_limit));
        }
		
        // truncated normal distribution
        double truncate(double x, double a, double b) {
            if (x >= a && x <= b) {
                return x;
            } else {
                return 0;
            }
        }
		
		
    protected :
        
        // ----- IMPLEMENTATION VARIABLES -----
        std::mt19937               random_engine;
        std::normal_distribution<> delay_random;
        std::normal_distribution<> weight_random;
        double                     weight_mu;
        double                     weight_sigma;
        double                     weight_lower_limit;
        double                     weight_upper_limit;
        double                     delay_mu;
        double                     delay_sigma;
        double                     delay_lower_limit;
        double                     delay_upper_limit;
	};
}

