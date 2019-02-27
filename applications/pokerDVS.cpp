/*
 * pokerDVS.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: Spiking neural network classifying the poker-DVS dataset
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/GUI/qtDisplay.hpp"

#include "../source/learningRules/timeInvariantSTDP.hpp"
#include "../source/learningRules/myelinPlasticity.hpp"

#include "../source/neurons/inputNeuron.hpp"
#include "../source/neurons/decisionMakingNeuron.hpp"
#include "../source/neurons/LIF.hpp"

#include "../source/addOns/spikeLogger.hpp"
#include "../source/addOns/myelinPlasticityLogger.hpp"

int main(int argc, char** argv) {
    //  ----- INITIALISING THE NETWORK -----
    hummus::QtDisplay qtDisplay;
    hummus::SpikeLogger spikeLog("spikeLog.bin");
    hummus::MyelinPlasticityLogger mpLog("mpLog.bin");

    hummus::Network network({&spikeLog, &mpLog}, &qtDisplay);

    //  ----- NETWORK PARAMETERS -----
	float eligibilityDecay = 100;
    bool overlappingRF = false;
    bool timeVaryingCurrent = true;
    bool homeostasis = true;
    bool wta = true;
    bool burst = false;
    
    //  ----- CREATING THE NETWORK -----
    hummus::MyelinPlasticity mp(1, 1, 1, 1);
    hummus::TimeInvariantSTDP t_stdp(1, -8, 3, 0);
    
    network.add2dLayer<hummus::InputNeuron>(0, 1, 34, 34, 1, overlappingRF, {});
    network.add2dLayer<hummus::LIF>(0, 1, 34, 34, 1, overlappingRF, {&mp}, timeVaryingCurrent, homeostasis, 10, 20, 3, wta, burst, eligibilityDecay);
    network.addDecisionMakingLayer<hummus::DecisionMakingNeuron>("../../data/pokerDVS/DHtrainLabel.txt", true, {&t_stdp}, 1000, timeVaryingCurrent, homeostasis, 10, 80, eligibilityDecay);

    //  ----- CONNECTING THE NETWORK -----
    network.allToAll(network.getLayers()[0], network.getLayers()[1], 0.006, 0.02, 10, 5);
    network.allToAll(network.getLayers()[1], network.getLayers()[2], 0.06, 0.02);

	//  ----- READING DATA FROM FILE -----
    hummus::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/pokerDVS/DHtrain.txt");
    auto testData = dataParser.readData("../../data/pokerDVS/DHtest.txt");

	//  ----- DISPLAY SETTINGS -----
	qtDisplay.useHardwareAcceleration(true);
	qtDisplay.setTimeWindow(5000);
	qtDisplay.trackLayer(1);
    qtDisplay.trackNeuron(network.getNeurons().back()->getNeuronID());

    //  ----- RUNNING THE NETWORK -----
    network.run(&trainingData, 1, &testData);

    //  ----- EXITING APPLICATION -----
    return 0;
}
