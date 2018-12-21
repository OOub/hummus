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

int main(int argc, char** argv)
{
    //  ----- INITIALISING THE NETWORK -----
	adonis_c::Analysis analysis("../../data/cards/test_nooff_pip4_rep1_jitter0Label.txt");
	adonis_c::Network network({&analysis});
	
    //  ----- NETWORK PARAMETERS -----
	float decayCurrent = 80;
	float decayPotential = 100;
	float refractoryPeriod = 3;
	float eligibilityDecay = 1000;
	
	bool overlap = false;
	bool homeostasis = true;
	bool wta = false;
	bool burst = true;
	
	//  ----- CREATING THE NETWORK -----
	adonis_c::STDP stdp;
	adonis_c::RewardModulatedSTDP rstdp;
	
	network.add2dLayer(1, 24, 24, {}, 1, -1, false, false, decayCurrent, decayPotential, refractoryPeriod, false, false, eligibilityDecay);
	network.add2dLayer(4, 24, 24, {&stdp}, 1, 1, overlap, homeostasis, decayCurrent, decayPotential, refractoryPeriod, wta, burst, eligibilityDecay);
	network.addDecisionMakingLayer("../../data/cards/train_nooff_pip4_rep50_jitter0Label.txt", {&rstdp}, 500);
	
	//  ----- CONNECTING THE NETWORK -----
	network.allToAll(network.getLayers()[0], network.getLayers()[1], 0.6, 0.3, 5, 3, 100);
	network.allToAll(network.getLayers()[1], network.getLayers()[2], 0.6, 0.3, 5, 3, 100);
	
	network.lateralInhibition(network.getLayers()[1], -1);
	
	//  ----- READING DATA FROM FILE -----
	adonis_c::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/cards/train_nooff_pip4_rep50_jitter0.txt");
	
	auto testData = dataParser.readData("../../data/cards/test_nooff_pip4_rep1_jitter0.txt");
	
    //  ----- RUNNING THE NETWORK -----
    network.run(&trainingData, &testData, 1);
	analysis.accuracy();
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
