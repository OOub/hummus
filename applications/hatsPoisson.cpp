/*
 * hatsNetwork.cpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 04/11/2018
 *
 * Information: spiking neural network running the n-Cars database with HATS encoded with the Poisson method;
 */

#include <iostream> 

#include "../source/network.hpp"
#include "../source/qtDisplay.hpp"
#include "../source/spikeLogger.hpp"
#include "../source/STDP.hpp"
#include "../source/analysis.hpp"

int main(int argc, char** argv)
{
    //  ----- INITIALISING THE NETWORK -----
	adonis_c::QtDisplay qtDisplay;
	adonis_c::Analysis analysis("../../data/hats/poisson/nCars_1samplePerc_1repLabel.txt");
	adonis_c::Network network({}, &qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	int gridWidth = 42;
	int gridHeight = 35;
	int rfSize = 7;
	
	float decayCurrent = 20;
	float decayPotential = 40;
	float refractoryPeriod = 3;
	float eligibilityDecay = 40;
	
	bool burst = false;
	bool homeostasis = false;
	bool wta = false;
	
	//  ----- INITIALISING THE LEARNING RULE -----
	adonis_c::STDP stdp;
	
	//  ----- CREATING THE NETWORK -----
	network.add2dLayer(rfSize, gridWidth, gridHeight, {}, 1, -1, false, false, decayCurrent, decayPotential, refractoryPeriod, false, false, eligibilityDecay);
	network.addLayer({&stdp}, 30, 1, 1, homeostasis, decayCurrent, decayPotential, refractoryPeriod, wta, burst, eligibilityDecay);
	network.addDecisionMakingLayer("../../data/hats/poisson/nCars_10samplePerc_1repLabel.txt", {}, decayCurrent, decayPotential, 100);
	
	network.allToAll(network.getLayers()[0], network.getLayers()[1], 1./20, 3);
	network.allToAll(network.getLayers()[1], network.getLayers()[2], 1./15, 3);
	
	//  ----- READING TRAINING DATA FROM FILE -----
	adonis_c::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/hats/poisson/nCars_10samplePerc_1rep.txt");
	
	//  ----- READING TEST DATA FROM FILE -----
	auto testData = dataParser.readData("../../data/hats/poisson/nCars_1samplePerc_1rep.txt");
	
    //  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(5000);
  	qtDisplay.trackLayer(2);
  	qtDisplay.trackOutputSublayer(0);
  	qtDisplay.trackNeuron(network.getNeurons().back().getNeuronID());
	
    //  ----- RUNNING THE NETWORK -----
    network.run(0.1, &trainingData, &testData);
	analysis.accuracy();
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
