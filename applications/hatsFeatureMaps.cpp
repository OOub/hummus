/*
 * hatsNetwork.cpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 09/10/2018
 *
 * Information: Spiking neural network running with histograms of averaged time surfaces converted into spikes.
 */

#include <iostream> 

#include "../source/network.hpp"
#include "../source/analysis.hpp"
#include "../source/qtDisplay.hpp"
#include "../source/predictionLogger.hpp"
#include "../source/rewardModulatedSTDP.hpp"
#include "../source/stdp.hpp"

int main(int argc, char** argv)
{
    //  ----- INITIALISING THE NETWORK -----
	adonis_c::QtDisplay qtDisplay;
	adonis_c::Analysis analysis("../../data/hats/testLabel.txt");
	adonis_c::Network network({&analysis}, &qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	float decayCurrent = 10;
	float decayPotential = 20;
	float refractoryPeriod = 0;
	float eligibilityDecay = 20;
	
	bool burst = false;
	bool wta = false;
	bool homeostasis = false;
	
	//  ----- INITIALISING THE LEARNING RULE -----
	adonis_c::STDP stdp;
	adonis_c::RewardModulatedSTDP rstdp;
	
	//  ----- CREATING THE NETWORK -----
	network.addLayer({}, 1470, 1, 6, false, decayCurrent, decayPotential, 0, false, false, eligibilityDecay);
	network.addLayer({&stdp}, 10, 1, 6, homeostasis, decayCurrent, decayPotential+20, refractoryPeriod, wta, burst, eligibilityDecay+20);
	network.addDecisionMakingLayer("../../data/hats/trainLabel.txt", {&rstdp}, 100);
	
	network.allToAll(network.getLayers()[0], network.getLayers()[1], 0.0006, 0.0004, 2, 0, 100);
	network.allToAll(network.getLayers()[1], network.getLayers()[2], 0.6, 0.4, 5, 3);
	network.lateralInhibition(network.getLayers()[1], -1);
	
	//  ----- READING TRAINING DATA FROM FILE -----
	adonis_c::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/hats/train.txt");
	
	//  ----- READING TEST DATA FROM FILE -----
	auto testData = dataParser.readData("../../data/hats/test.txt");
	
	//  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(1000);
  	qtDisplay.trackLayer(2);
	qtDisplay.trackNeuron(network.getNeurons().back().getNeuronID());
	
    //  ----- RUNNING THE NETWORK -----
    network.run(&trainingData, &testData, 0.5);
	analysis.accuracy();
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
