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
#include "../source/randomDistributions/normal.hpp"
#include "../source/GUI/qtDisplay.hpp"

#include "../source/learningRules/timeInvariantSTDP.hpp"

#include "../source/neurons/LIF.hpp"
#include "../source/neurons/inputNeuron.hpp"
#include "../source/neurons/decisionMakingNeuron.hpp"

#include "../source/addOns/spikeLogger.hpp"

int main(int argc, char** argv) {
    //  ----- INITIALISING THE NETWORK -----
    hummus::QtDisplay qtDisplay;

    hummus::Network network(&qtDisplay);

    //  ----- NETWORK PARAMETERS -----
    bool timeVaryingCurrent = false;
    bool homeostasis = false;
    bool wta = true;
    
    //  ----- CREATING THE NETWORK -----
    auto ti_stdp = network.makeLearningRule<hummus::TimeInvariantSTDP>();
    
    network.add2dLayer<hummus::InputNeuron>(34, 34, 1, {});
    network.addConvolutionalLayer<hummus::LIF>(network.getLayers()[0], 5, 1, hummus::Rand(), 100, 1, {ti_stdp}, timeVaryingCurrent, homeostasis, 10, 20, 3, wta);
    network.addPoolingLayer<hummus::LIF>(network.getLayers()[1], hummus::Rand(), 100, {}, timeVaryingCurrent, homeostasis, 10, 20, 3, wta);
    network.addDecisionMakingLayer<hummus::DecisionMakingNeuron>("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/pokerDVS/DHtrainingLabel.txt");
    
    //  ----- CONNECTING THE NETWORK -----
    network.allToAll(network.getLayers()[2], network.getLayers()[3], hummus::Normal());
    
	//  ----- READING DATA FROM FILE -----
    hummus::DataParser dataParser;
    auto trainingData = dataParser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/pokerDVS/DHtraining.txt");
    auto testData = dataParser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/pokerDVS/DHtest.txt");

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
