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
#include "../source/stdp.hpp"
#include "../source/rewardModulatedSTDP.hpp"
#include "../source/myelinPlasticity.hpp"
#include "../source/predictionLogger.hpp"

int main(int argc, char** argv)
{
    //  ----- INITIALISING THE NETWORK -----
    adonis_c::QtDisplay qtDisplay;
    adonis_c::PredictionLogger predictionLogger("predictionLogger.bin");
	adonis_c::Analysis analysis("../../data/cards/testLabel.txt");
	adonis_c::Network network({&analysis}, &qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	float decayCurrent = 80;
	float decayPotential = 100;
	float refractoryPeriod = 6000;
	float eligibilityDecay = 100;
	
	bool overlap = false;
	bool homeostasis = true;
	bool wta = true;
	bool burst = false;
	
	//  ----- CREATING THE NETWORK -----
	adonis_c::STDP stdp(1, 1, 100, 100);
	adonis_c::MyelinPlasticity mp(0.1, 0.1, false);
	
	network.add2dLayer(17, 34, 34, {}, 1, -1, false, homeostasis, decayCurrent, decayPotential, 3, wta, burst, eligibilityDecay);
	network.add2dLayer(17, 34, 34, {}, 1, 100, overlap, homeostasis, decayCurrent, decayPotential, refractoryPeriod, wta, burst, eligibilityDecay+10);
	network.addDecisionMakingLayer("../../data/cards/trainLabel.txt", {&stdp, &mp}, 6000, false, decayCurrent, decayPotential, true, burst, eligibilityDecay+50);

	//  ----- CONNECTING THE NETWORK -----
	network.allToAll(network.getLayers()[0], network.getLayers()[1], 0.05, 0.1, 0, 0, 50);
	network.allToAll(network.getLayers()[1], network.getLayers()[2], 1./4, 0, 100, 60, 50);
	
	network.lateralInhibition(network.getLayers()[1], -1);
	
	//  ----- READING DATA FROM FILE -----
	adonis_c::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/cards/train.txt");
	
	auto testData = dataParser.readData("../../data/cards/test.txt");
	
	//  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(20000);
  	qtDisplay.trackLayer(1);
  	std::cout << "last neuron " << network.getNeurons().back().getNeuronID() << std::endl;
	qtDisplay.trackNeuron(network.getNeurons().back().getNeuronID());
	
    //  ----- RUNNING THE NETWORK -----
    network.run(&trainingData, &testData, 5);
	analysis.accuracy();
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
