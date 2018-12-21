/*
 * hatsNetwork.cpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 04/11/2018
 *
 * Information: spiking neural network running the n-Cars database with HATS encoded with the Intentisty-to-latency method;
 */

#include <iostream> 

#include "../source/GUI/qtDisplay.hpp"
#include "../source/core.hpp"
#include "../source/addOns/predictionLogger.hpp"
#include "../source/addOns/analysis.hpp"
#include "../source/learningRules/stdp.hpp"
#include "../source/learningRules/rewardModulatedSTDP.hpp"

int main(int argc, char** argv)
{
    //  ----- INITIALISING THE NETWORK -----
    adonis::QtDisplay qtDisplay;
	adonis::PredictionLogger predictionLogger("hatsLatency.bin");
	adonis::Analysis analysis("../../data/hats/latency/test_nCars_10samplePerc_1repLabel.txt");
	adonis::Network network({&predictionLogger, &analysis}, &qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	float decayCurrent = 10;
	float decayPotential = 20;
	float refractoryPeriod = 3;
	float eligibilityDecay = 100;
	
	bool burstingActivity = false;
	bool homeostasis = false;
	bool wta = true;
	
	//  ----- INITIALISING THE LEARNING RULE -----
	adonis::STDP stdp;
	
	//  ----- CREATING THE NETWORK -----
	network.addLayer({}, 4116, 1, 1, false, decayCurrent, decayPotential, refractoryPeriod, false, false, eligibilityDecay);
	network.addLayer({&stdp}, 10, 1, 1, homeostasis, decayCurrent, decayPotential, refractoryPeriod, wta, burstingActivity, eligibilityDecay);
	network.addDecisionMakingLayer("../../data/hats/latency/train_nCars_10samplePerc_1repLabel.txt", {});
	
	network.allToAll(network.getLayers()[0], network.getLayers()[1], 0.6, 0.4, 5, 3);
	network.allToAll(network.getLayers()[1], network.getLayers()[2], 0.6, 0.4, 5, 3);
	
	//  ----- READING TRAINING DATA FROM FILE -----
	adonis::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/hats/latency/train_nCars_10samplePerc_1rep.txt");
	
	//  ----- READING TEST DATA FROM FILE -----
	auto testingData = dataParser.readData("../../data/hats/latency/test_nCars_10samplePerc_1rep.txt");
	
	//  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(5000);
  	qtDisplay.trackLayer(2);
	qtDisplay.trackNeuron(network.getNeurons().back().getNeuronID());
	
    //  ----- RUNNING THE NETWORK -----
    network.run(1, &trainingData, &testingData);
	analysis.accuracy();
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
