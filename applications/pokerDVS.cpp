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

    hummus::Network network(&qtDisplay);

    //  ----- NETWORK PARAMETERS -----
    bool timeVaryingCurrent = false;
    bool homeostasis = false;
    bool wta = true;
    bool burst = false;
    
    //  ----- CREATING THE NETWORK -----
    hummus::MyelinPlasticity mp(1, 1, 1, 1);
    hummus::TimeInvariantSTDP t_stdp(1, -8, 3, 0);
    
    network.add2dLayer<hummus::InputNeuron>(34, 34, 1, {});
//    network.addLayer<hummus::LIF>(34*34, 1, 1, {&mp}, timeVaryingCurrent, homeostasis, 10, 20, 3, wta, burst, 20);
//    network.addDecisionMakingLayer<hummus::DecisionMakingNeuron>("../../data/pokerDVS/DHtrainingLabel.txt", true, {&t_stdp}, 2000, timeVaryingCurrent, homeostasis, 10, 80, 80);
    
    //  ----- CONNECTING THE NETWORK -----
//    network.allToAll(network.getLayers()[0], network.getLayers()[1], 0.01, 0.005, 5, 3);
//    network.allToAll(network.getLayers()[1], network.getLayers()[2], 0.1, 0.05, 5, 3);
//    std::cout << "connection done" << std::endl;
    
	//  ----- READING DATA FROM FILE -----
    hummus::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/pokerDVS/DHtraining.txt");
    auto testData = dataParser.readData("../../data/pokerDVS/DHtest.txt");

	//  ----- DISPLAY SETTINGS -----
	qtDisplay.useHardwareAcceleration(true);
	qtDisplay.setTimeWindow(5000);
	qtDisplay.trackLayer(1);
    qtDisplay.trackNeuron(network.getNeurons().back()->getNeuronID());

    //  ----- RUNNING THE NETWORK -----
    network.run(&trainingData, 0, &testData);

    //  ----- EXITING APPLICATION -----
    return 0;
}
