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
#include "../source/STDP.hpp"
#include "../source/rewardModulatedSTDP.hpp"

int main(int argc, char** argv)
{
    //  ----- INITIALISING THE NETWORK -----
	adonis_c::QtDisplay qtDisplay;
	adonis_c::Analysis analysis("../../data/cards/test_pip4_rep10_jitter0Label.txt");
	adonis_c::Network network({&analysis}, &qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	float decayCurrent = 10;
	float decayPotential = 20;
	float refractoryPeriod = 3;
	float eligibilityDecay = 100;
	
	bool overlap = true;
	bool homeostasis = true;
	bool wta = false;
	bool burst = false;
	
	//  ----- CREATING THE NETWORK -----
	adonis_c::STDP stdp;
	adonis_c::RewardModulatedSTDP rstdp;
	
	network.add2dLayer(4, 24, 24, {}, 1, -1, false, false, decayCurrent, decayPotential, refractoryPeriod, false, false, eligibilityDecay);
	network.add2dLayer(4, 24, 24, {&rstdp}, 1, 1, overlap, homeostasis, decayCurrent+10, decayPotential+10, refractoryPeriod, wta, burst, eligibilityDecay+10);
	network.addDecisionMakingLayer("../../data/cards/train_pip4_rep10_jitter0Label.txt", {});

	//  ----- CONNECTING THE NETWORK -----
	network.convolution(network.getLayers()[0], network.getLayers()[1], 0.5, 1, 20, 5);
	network.allToAll(network.getLayers()[1], network.getLayers()[2], 0.5, 1, 20, 5);
	
	network.lateralInhibition(network.getLayers()[1], -1);
	
	//  ----- READING DATA FROM FILE -----
	adonis_c::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/cards/train_pip4_rep10_jitter0.txt");
	auto testData = dataParser.readData("../../data/cards/train_pip4_rep10_jitter0.txt");

	//  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(5000);
  	qtDisplay.trackLayer(2);
	qtDisplay.trackNeuron(network.getNeurons().back().getNeuronID());
	
    //  ----- RUNNING THE NETWORK -----
    network.run(0.1, &trainingData, &testData);
	analysis.accuracy();
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
