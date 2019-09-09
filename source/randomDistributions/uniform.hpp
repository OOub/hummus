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
	
	enum class uniform_type {
		integer,
		real
	};
	
	class Uniform {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        Uniform(float weight_lower_limit=0, float weight_upper_limit=1, float delay_lower_limit=0, float delay_upper_limit=0, uniform_type _int_or_real=uniform_type::integer) :
        		int_or_real(_int_or_real) {
			
			// error handling
                    if (delay_lower_limit < 0 || delay_upper_limit < 0) {
				throw std::logic_error("the delays cannot be in a negative range");
			}

            // randomising weights and delays
            std::random_device device;
            random_engine = std::mt19937(device());
			
			if (_int_or_real == uniform_type::integer) {
                int_delay_random = std::uniform_int_distribution<>(static_cast<int>(delay_lower_limit), static_cast<int>(delay_upper_limit));
                int_weight_random = std::uniform_int_distribution<>(static_cast<int>(weight_lower_limit), static_cast<int>(weight_upper_limit));
			} else {
                real_delay_random = std::uniform_real_distribution<>(delay_lower_limit, delay_upper_limit);
                real_weight_random = std::uniform_real_distribution<>(weight_lower_limit, weight_upper_limit);
			}
        }
		
		
        std::pair<float, float> operator()(int16_t x, int16_t y, int16_t depth) {
        	if (int_or_real == uniform_type::integer) {
				return std::make_pair(real_weight_random(random_engine), real_delay_random(random_engine));
			} else {
				return std::make_pair(int_weight_random(random_engine), int_delay_random(random_engine));
			}
        }
		
    protected :
        
        // ----- IMPLEMENTATION VARIABLES -----
        std::mt19937                     random_engine;
        std::uniform_int_distribution<>  int_delay_random;
		std::uniform_real_distribution<> real_delay_random;
        std::uniform_int_distribution<>  int_weight_random;
		std::uniform_real_distribution<> real_weight_random;
		uniform_type                     int_or_real;
	};
}

