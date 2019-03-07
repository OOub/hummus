/*
 * jsonParser.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/05/2018
 *
 * Information: Example of a network being build from a JSON save file
 */

#include <string>
#include <iostream>

#include "../source/builder.hpp"
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

#include "../source/dependencies/json.hpp"

int main(int argc, char** argv) {
    hummus::QtDisplay qtDisplay;
    hummus::Network network(&qtDisplay);

    hummus::Builder bob(&network);
    bob.import("../../data/testSave.json");
    
    return 0;
}
