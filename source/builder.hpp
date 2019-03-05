/*
 * builder.hpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 05/02/2019
 *
 * Information: the Builder class takes a network instance and allows us to use the import method to build the network from a saved network JSON file
 */

#pragma once

#include "core.hpp"
#include "rand.hpp"

#include "neurons/inputNeuron.hpp"
#include "neurons/decisionMakingNeuron.hpp"
#include "neurons/LIF.hpp"
#include "neurons/IF.hpp"

#include "learningRules/stdp.hpp"
#include "learningRules/timeInvariantSTDP"
#include "learningRules/myelinPlasticity.hpp"
#include "learningRules/rewardModulatedSTDP.hpp"

#include "dependencies/json.hpp"

namespace hummus {
    
    class Network;
    
	class Builder {
        
	public:
		// ----- CONSTRUCTOR AND DESTRUCTOR -----
        Builder(Network* _network) :
                network(_network) {}
        
		virtual ~Builder(){}
		
		// ----- PUBLIC METHODS -----
        // importing a previously created network from a JSON file
        void import(std::string filename) {
            nlohmann::json input;
            std::ifstream input_file(filename);
            input_file >> input;
            
            if (input.is_array()) {
                if (input[0].is_object()){
                    // do stuff
                }
            } else {
                throw std::logic_error("the imported file does not correspond to a saved network JSON file");
            }
        }
        
    private:
        
        // ----- IMPLEMENTATON VARIABLES -----
        Network* network;
	};
}
