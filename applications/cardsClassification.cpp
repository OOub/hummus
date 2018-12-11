/*
 * cardsClassification.cpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 11/12/2018
 *
 * Information: Spiking neural network classifying the poker-DVS dataset
 */

#include <iostream>

#include "../source/network.hpp"
#include "../source/qtDisplay.hpp"

int main(int argc, char** argv)
{
    //  ----- INITIALISING THE NETWORK -----
	adonis_c::QtDisplay qtDisplay;
	adonis_c::Network network(&qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	
    // IDs for each layer (order is important)
    int layer0 = 0;
    int layer1 = 1;
    int layer2 = 2;
	
	int gridWidth = 24;
	int gridHeight = 24;
	int rfSize = 4;
	
	float decayCurrent = 10;
	float decayPotential = 20;
	float refractoryPeriod = 3;
	bool  burstingActivity = false;
	float eligibilityDecay = 20;
	
	//  ----- CREATING THE NETWORK -----
	network.add2dLayer(layer0, rfSize, gridWidth, gridHeight, {}, 1, -1, false, decayCurrent, decayPotential, refractoryPeriod, burstingActivity, eligibilityDecay);
	network.add2dLayer(layer1, rfSize, gridWidth, gridHeight, {}, 1, 1, false, decayCurrent, decayPotential, refractoryPeriod, burstingActivity, eligibilityDecay);
	network.addDecisionMakingLayer(layer2, "../../data/cards/test_pip4_rep10_jitter0Label.txt", {});

	network.convolution(network.getLayers()[layer0], network.getLayers()[layer1], true, 1., true, 20);
	network.allToAll(network.getLayers()[layer1], network.getLayers()[layer2], true, 1., true, 20);
	
	//  ----- READING DATA FROM FILE -----
	adonis_c::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/cards/train_pip4_rep10_jitter0.txt");
	auto testData = dataParser.readData("../../data/cards/test_pip4_rep10_jitter0.txt");

	//  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(5000);
  	qtDisplay.trackLayer(2);
	qtDisplay.trackNeuron(network.getNeurons().back().getNeuronID());
	
    //  ----- RUNNING THE NETWORK -----
    network.run(0.1, &trainingData, &testData);
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
