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
#include "../source/analysis.hpp"
#include "../source/qtDisplay.hpp"
#include "../source/myelinPlasticity.hpp"
#include "../source/predictionLogger.hpp"

int main(int argc, char** argv)
{
    //  ----- INITIALISING THE NETWORK -----
    adonis_c::QtDisplay qtDisplay;
	adonis_c::Network network(&qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	float decayCurrent = 10;
	float decayPotential = 20;
	float eligibilityDecay = 20;
	
	bool homeostasis = false;
	bool wta = true;
	bool burst = false;
	
	//  ----- CREATING THE NETWORK -----
	adonis_c::MyelinPlasticity mp(1, 1, true);
	
	network.add2dLayer(1, 34, 34, {}, 1, -1, false, homeostasis, decayCurrent, decayPotential, 0, wta, false, eligibilityDecay);
	network.addDecisionMakingLayer("../../data/cards/trainLabel.txt", {&mp}, 900, false, decayCurrent, decayPotential, wta, burst, eligibilityDecay);

	//  ----- CONNECTING THE NETWORK -----
	network.allToAll(network.getLayers()[0], network.getLayers()[1], 0.03, 0.02, 5, 3, 100);
//	network.lateralInhibition(network.getLayers()[0], -0.5);
	
	//  ----- READING DATA FROM FILE -----
	adonis_c::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/cards/train.txt");
	
	auto testData = dataParser.readData("../../data/cards/test.txt");
	
	//  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(10000);
  	qtDisplay.trackLayer(1);
	qtDisplay.trackNeuron(network.getNeurons().back().getNeuronID());
	
    //  ----- RUNNING THE NETWORK -----
    network.run(&trainingData, &testData, 1);
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
