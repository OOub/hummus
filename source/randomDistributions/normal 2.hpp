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
        Normal(float _weightMu=1, float _weightSigma=0, float _delayMu=0, float _delaySigma=0, float _weightLowerLimit=-INFINITY, float _weightUpperLimit=INFINITY, float _delayLowerLimit=-INFINITY, float _delayUpperLimit=INFINITY) :
                weightMu(_weightMu),
                weightSigma(_weightSigma),
                weightLowerLimit(_weightLowerLimit),
                weightUpperLimit(_weightUpperLimit),
                delayMu(_delayMu),
                delaySigma(_delaySigma),
                delayLowerLimit(_delayLowerLimit),
                delayUpperLimit(_delayUpperLimit) {
            // randomising weights and delays
            std::random_device device;
            randomEngine = std::mt19937(device());
            delayRandom = std::normal_distribution<>(delayMu, delaySigma);
            weightRandom = std::normal_distribution<>(weightMu, weightSigma);
        }
		
        std::pair<float, float> operator()(int16_t x, int16_t y, int16_t depth) {
            return std::make_pair(truncate(weightRandom(randomEngine), weightLowerLimit, weightUpperLimit), std::abs(truncate(delayRandom(randomEngine), delayLowerLimit, delayUpperLimit)));
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
        std::mt19937               randomEngine;
        std::normal_distribution<> delayRandom;
        std::normal_distribution<> weightRandom;
        double                     weightMu;
        double                     weightSigma;
        double                     weightLowerLimit;
        double                     weightUpperLimit;
        double                     delayMu;
        double                     delaySigma;
        double                     delayLowerLimit;
        double                     delayUpperLimit;
	};
}

