/*
 * cardsClassification.cpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 11/12/2018
 *
 * Information: Spiking neural network classifying the poker-DVS dataset
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/addOns/analysis.hpp"
#include "../source/GUI/qtDisplay.hpp"
#include "../source/learningRules/stdp.hpp"
#include "../source/learningRules/rewardModulatedSTDP.hpp"
#include "../source/learningRules/myelinPlasticity.hpp"

int main(int argc, char** argv)
{
    //  ----- INITIALISING THE NETWORK -----
    adonis::QtDisplay qtDisplay;
	adonis::Analysis analysis("../../data/cards/test_nooff_pip2_rep10_jitter0Label.txt");
	adonis::Network network({&analysis}, &qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	float decayCurrent = 10;
	float decayPotential = 20;
	float refractoryPeriod = 3;
	float eligibilityDecay = 100;
	
	bool overlap = false;
	bool homeostasis = false;
	bool wta = false;
	bool burst = false;
	
	//  ----- CREATING THE NETWORK -----
	adonis::STDP stdp;
	adonis::RewardModulatedSTDP rstdp;
	
	network.add2dLayer(1, 24, 24, {}, 1, -1, false, false, decayCurrent, decayPotential, refractoryPeriod, false, false, eligibilityDecay);
	network.add2dLayer(4, 24, 24, {}, 1, 1, overlap, homeostasis, decayCurrent, decayPotential, refractoryPeriod, wta, burst, eligibilityDecay);
	network.addDecisionMakingLayer("../../data/cards/train_nooff_pip2_rep50_jitter0Label.txt", {}, 500);

	//  ----- CONNECTING THE NETWORK -----
	network.allToAll(network.getLayers()[0], network.getLayers()[1], 0.6, 0.4, 5, 3, 50);
	network.allToAll(network.getLayers()[1], network.getLayers()[2], 0.6, 0.4, 5, 3, 50);
	
	network.lateralInhibition(network.getLayers()[1], -1);
	network.lateralInhibition(network.getLayers()[2], -1);
	
	//  ----- READING DATA FROM FILE -----
	adonis::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/cards/train_nooff_pip2_rep50_jitter0.txt");
	auto testData = dataParser.readData("../../data/cards/test_nooff_pip2_rep10_jitter0.txt");
	
	//  ----- DISPLAY SETTINGS -----
	qtDisplay.useHardwareAcceleration(true);
	qtDisplay.setTimeWindow(5000);
	qtDisplay.trackLayer(2);
	
    //  ----- RUNNING THE NETWORK -----
    network.run(0.1, &trainingData, &testData);
	analysis.accuracy();
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
