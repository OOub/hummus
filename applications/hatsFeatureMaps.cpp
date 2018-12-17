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
#include "../source/STDP.hpp"

int main(int argc, char** argv)
{
    //  ----- INITIALISING THE NETWORK -----
	adonis_c::QtDisplay qtDisplay;
	adonis_c::Analysis analysis("../../data/hats/feature_maps/nCars_100samplePerc_1repLabel.txt");
	adonis_c::PredictionLogger predictionLogger("hatsFeatureMaps.bin");
	adonis_c::Network network({&predictionLogger, &analysis});
	
    //  ----- NETWORK PARAMETERS -----
	
	int gridWidth = 42;
	int gridHeight = 35;
	int rfSize = 7;
	
	float decayCurrent = 10;
	float decayPotential = 20;
	float refractoryPeriod = 3;
	float eligibilityDecay = 20;
	
	bool burst = false;
	bool overlap = false;
	bool wta = false;
	bool homeostasis = false;
	
	//  ----- INITIALISING THE LEARNING RULE -----
	adonis_c::STDP stdp;
	adonis_c::RewardModulatedSTDP rstdp;
	
	//  ----- CREATING THE NETWORK -----
	network.add2dLayer(rfSize, gridWidth, gridHeight, {}, 3, -1, false, false, decayCurrent, decayPotential, refractoryPeriod, false, false, eligibilityDecay);
	network.add2dLayer(rfSize, gridWidth, gridHeight, {}, 1, 1, overlap, homeostasis, decayCurrent, decayPotential, refractoryPeriod, wta, burst, eligibilityDecay);
	network.addDecisionMakingLayer("../../data/hats/feature_maps/nCars_100samplePerc_10repLabel.txt", {}, decayCurrent, decayPotential, 1000);

	network.convolution(network.getLayers()[0], network.getLayers()[1], 1., 0, 20, 0);
	network.allToAll(network.getLayers()[1], network.getLayers()[2], 1., 0, 20, 0);
	
	//  ----- READING TRAINING DATA FROM FILE -----
	adonis_c::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/hats/feature_maps/nCars_100samplePerc_10rep.txt");
	
	//  ----- READING TEST DATA FROM FILE -----
	auto testData = dataParser.readData("../../data/hats/feature_maps/nCars_100samplePerc_1rep.txt");
	
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
