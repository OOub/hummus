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
#include "../source/randomDistributions/normal.hpp"
#include "../source/GUI/qt/qtDisplay.hpp"

#include "../source/neurons/input.hpp"
#include "../source/neurons/decisionMaking.hpp"
#include "../source/neurons/LIF.hpp"
#include "../source/neurons/IF.hpp"

#include "../source/addons/spikeLogger.hpp"
#include "../source/addons/potentialLogger.hpp"
#include "../source/addons/classificationLogger.hpp"
#include "../source/addons/myelinPlasticityLogger.hpp"
#include "../source/addons/analysis.hpp"

#include "../source/learningRules/myelinPlasticity.hpp"
#include "../source/learningRules/rewardModulatedSTDP.hpp"
#include "../source/learningRules/stdp.hpp"

#include "../source/synapticKernels/exponential.hpp"
#include "../source/synapticKernels/dirac.hpp"
#include "../source/synapticKernels/step.hpp"

#include "../source/dependencies/json.hpp"

int main(int argc, char** argv) {
    hummus::Network network;
    auto& display = network.makeGUI<hummus::QtDisplay>();
    
    hummus::Builder bob(&network);
    bob.import("../../data/testSave.json");
    
    //  ----- INJECTING SPIKES -----
    network.injectSpike(0, 10);
    network.injectSpike(0, 11);
    network.injectSpike(0, 30);
    
    //  ----- DISPLAY SETTINGS -----
    display.setTimeWindow(100);
    display.trackNeuron(1);
    
    network.run(100, 0.1);
    
    return 0;
}
