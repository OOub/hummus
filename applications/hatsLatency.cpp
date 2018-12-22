/*
 * hatsNetwork.cpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 04/11/2018
 *
 * Information: spiking neural network running the n-Cars database with HATS encoded with the Intentisty-to-latency method;
 */

#include <iostream> 

#include "../source/qtDisplay.hpp"
#include "../source/network.hpp"
#include "../source/predictionLogger.hpp"
#include "../source/analysis.hpp"
#include "../source/stdp.hpp"
#include "../source/rewardModulatedSTDP.hpp"

int main(int argc, char** argv)
{
    //  ----- INITIALISING THE NETWORK -----
    adonis_c::QtDisplay qtDisplay;
	adonis_c::PredictionLogger predictionLogger("hatsLatency.bin");
	adonis_c::Analysis analysis("../../data/hats/testLabel.txt");
	adonis_c::Network network({&predictionLogger, &analysis}, &qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	float decayCurrent = 10;
	float decayPotential = 20;
	float refractoryPeriod = 3;
	float eligibilityDecay = 100;
	
	bool burstingActivity = false;
	bool homeostasis = true;
	bool wta = true;
	
	//  ----- INITIALISING THE LEARNING RULE -----
	adonis_c::STDP stdp(0.1, 0.1, 100, 100);
	adonis_c::RewardModulatedSTDP rstdp;
	
	//  ----- CREATING THE NETWORK -----
	network.addLayer({}, 1470, 1, 1, false, decayCurrent, decayPotential, refractoryPeriod, false, false, eligibilityDecay);
	network.addLayer({&stdp}, 10, 1, 1, homeostasis, decayCurrent, decayPotential, refractoryPeriod, wta, burstingActivity, eligibilityDecay);
	network.addDecisionMakingLayer("../../data/hats/trainLabel.txt", {}, 900, false, decayCurrent, decayPotential);
	
	network.allToAll(network.getLayers()[0], network.getLayers()[1], 0.2, 0.4, 5, 3, 100);
	network.allToAll(network.getLayers()[1], network.getLayers()[2], 0.6, 0.4);
//	network.lateralInhibition(network.getLayers()[1], -1);
	
	//  ----- READING TRAINING DATA FROM FILE -----
	adonis_c::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/hats/train.txt");
	
	//  ----- READING TEST DATA FROM FILE -----
	auto testingData = dataParser.readData("../../data/hats/test.txt");
	
	//  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(5000);
  	qtDisplay.trackLayer(1);
	qtDisplay.trackNeuron(network.getNeurons().back().getNeuronID());
	
    //  ----- RUNNING THE NETWORK -----
    network.run(&trainingData, &testingData, 10);
	analysis.accuracy();
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
