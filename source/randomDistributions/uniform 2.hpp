/*
 * uniform.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 28/02/2019
 *
 * Information: The UniformInt class can be used as an input for network methods that require lambda function to connect layers with weights and delays following a particular distribution (eg. allToAll). In this case, the distribution is either a uniform_int_distribution or a uniform_real_distribution depedning on the chosen enum
 */

#pragma once

#include <random>
#include <cmath>

namespace hummus {
	
	enum class uniformType {
		integerType,
		realType
	};
	
	class Uniform {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        Uniform(float weightLowerLimit=0, float weightUpperLimit=1, float delayLowerLimit=0, float delayUpperLimit=0, uniformType _int_or_real=uniformType::integerType) :
        		int_or_real(_int_or_real) {
			
			// error handling
			if (delayLowerLimit < 0 || delayUpperLimit < 0) {
				throw std::logic_error("the delays cannot be in a negative range");
			}

            // randomising weights and delays
            std::random_device device;
            randomEngine = std::mt19937(device());
			
			if (_int_or_real == uniformType::integerType) {
            	int_delayRandom = std::uniform_int_distribution<>(static_cast<int>(delayLowerLimit), static_cast<int>(delayUpperLimit));
            	int_weightRandom = std::uniform_int_distribution<>(static_cast<int>(weightLowerLimit), static_cast<int>(weightUpperLimit));
			} else {
				real_delayRandom = std::uniform_real_distribution<>(delayLowerLimit, delayUpperLimit);
				real_weightRandom = std::uniform_real_distribution<>(weightLowerLimit, weightUpperLimit);
			}
        }
		
		
        std::pair<float, float> operator()(int16_t x, int16_t y, int16_t depth) {
        	if (int_or_real == uniformType::integerType) {
				return std::make_pair(real_weightRandom(randomEngine), real_delayRandom(randomEngine));
			} else {
				return std::make_pair(int_weightRandom(randomEngine), int_delayRandom(randomEngine));
			}
        }
		
    protected :
        
        // ----- IMPLEMENTATION VARIABLES -----
        std::mt19937                     randomEngine;
        std::uniform_int_distribution<>  int_delayRandom;
		std::uniform_real_distribution<> real_delayRandom;
        std::uniform_int_distribution<>  int_weightRandom;
		std::uniform_real_distribution<> real_weightRandom;
		uniformType                      int_or_real;
	};
}

