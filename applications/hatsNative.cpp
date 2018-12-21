/*
 * hatsNetwork.cpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 09/10/2018
 *
 * Information: spiking neural network running the n-Cars database
 */

#include <iostream> 

#include "../source/network.hpp"
#include "../source/qtDisplay.hpp"
#include "../source/predictionLogger.hpp"
#include "../source/analysis.hpp"
#include "../source/rewardModulatedSTDP.hpp"
#include "../source/stdp.hpp"
#include "../source/myelinPlasticity.hpp"

int main(int argc, char** argv)
{
    //  ----- INITIALISING THE NETWORK -----
    adonis::Analysis analysis("../../data/hats/native/nCars_100samplePerc_1repLabel.txt");
	adonis::PredictionLogger predictionLogger("hatsNative.bin");
	adonis::Network network({&analysis, &predictionLogger});
	
    //  ----- NETWORK PARAMETERS -----
	int gridWidth = 60;
	int gridHeight = 50;
	int rfSize = 10;
	
	float decayCurrent = 10;
	float decayPotential = 20;
	float refractoryPeriod = 3;
	float eligibilityDecay = 20;
	
	bool burst = false;
	bool wta = false;
	bool homeostasis = false;
	bool overlap = false;
	
	//  ----- INITIALISING THE LEARNING RULE -----
	adonis::STDP stdp;
	adonis::MyelinPlasticity myelinPlasticity(1, 1);
	adonis::RewardModulatedSTDP rstdp;
	
	//  ----- CREATING THE NETWORK -----
	network.add2dLayer(rfSize, gridWidth, gridHeight, {}, 1, -1, false, false, decayCurrent, decayPotential, refractoryPeriod, false, false, eligibilityDecay);
	network.add2dLayer(rfSize, gridWidth, gridHeight, {&stdp}, 1, 1, overlap, homeostasis, decayCurrent+10, decayPotential+10, refractoryPeriod, wta, burst, eligibilityDecay);
	network.addDecisionMakingLayer("../../data/hats/native/nCars_100samplePerc_10repLabel.txt", {});
	
	network.allToAll(network.getLayers()[0], network.getLayers()[1], 0.5, 0.5);
	network.allToAll(network.getLayers()[1], network.getLayers()[2], 0.5, 0.5);
	
	//  ----- READING TRAINING DATA FROM FILE -----
	adonis::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/hats/native/nCars_100samplePerc_10rep.txt");
	
	//  ----- READING TEST DATA FROM FILE -----
	auto testData = dataParser.readData("../../data/hats/native/nCars_100samplePerc_1rep.txt");
	
	//  ----- INJECTING TEST SPIKES -----
	network.injectSpikeFromData(&testData);
	
    //  ----- RUNNING THE NETWORK -----
    network.run(0.5, &trainingData, &testData);
	analysis.accuracy();
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
