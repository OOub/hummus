/*
 * jsonParser.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/05/2018
 *
 * Information: Example of a basic spiking neural network.
 */

#include <string>
#include <iostream>

#include "../source/core.hpp"
#include "../source/rand.hpp"
#include "../source/GUI/qtDisplay.hpp"

#include "../source/neurons/inputNeuron.hpp"
#include "../source/neurons/decisionMakingNeuron.hpp"
#include "../source/neurons/LIF.hpp"
#include "../source/neurons/IF.hpp"

#include "../source/addOns/spikeLogger.hpp"
#include "../source/addOns/potentialLogger.hpp"
#include "../source/addOns/classificationLogger.hpp"
#include "../source/addOns/myelinPlasticityLogger.hpp"
#include "../source/addOns/analysis.hpp"

#include "../source/learningRules/myelinPlasticity.hpp"
#include "../source/learningRules/rewardModulatedSTDP.hpp"
#include "../source/learningRules/stdp.hpp"

using json = nlohmann::json;

int main(int argc, char** argv) {
//	std::ifstream input;
//	std::string filename;
//	if (argc == 2)
//	{
//		filename = argv[1];
//	}
//	else if (argc == 1)
//	{
//		std::cout << "enter a file name: " << std::endl;
//		std::cin >> filename;
//	}
//	input.open(filename.c_str());
//	if (input.fail())
//	{
//		throw std::logic_error(filename.append(" could not be opened"));
//	}
//	
//	std::cout << "yes" << std::endl;
//    //  ----- EXITING APPLICATION -----
    return 0;
}
