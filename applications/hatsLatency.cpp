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
#include "../source/myelinPlasticity.hpp"

int main(int argc, char** argv)
{
    //  ----- INITIALISING THE NETWORK -----
    adonis_c::QtDisplay qtDisplay;
    adonis_c::SpikeLogger spikeLog("spikeLog.bin");
    adonis_c::PredictionLogger predictionLog("predictionLog.bin");
	adonis_c::Analysis analysis("../../data/hats/testLabel2.txt");
	adonis_c::Network network({&analysis});
	
    //  ----- NETWORK PARAMETERS -----
	float decayCurrent = 10;
	float decayPotential = 20;
	float eligibilityDecay = 20;
	
	//  ----- INITIALISING THE LEARNING RULE -----
	adonis_c::MyelinPlasticity mp(0.1, 0.1, true);
	
	//  ----- CREATING THE NETWORK -----
	network.addLayer({}, 1470, 1, 1, false, decayCurrent, decayPotential, 0, false, false, eligibilityDecay);
	network.addDecisionMakingLayer("../../data/hats/trainLabel2.txt", {&mp}, 900, false, decayCurrent, decayPotential, true, false, eligibilityDecay+50);
	
	network.allToAll(network.getLayers()[0], network.getLayers()[1], 0.05, 0.02, 5, 3, 100);
	
	//  ----- READING TRAINING DATA FROM FILE -----
	adonis_c::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/hats/train2.txt");
	
	//  ----- READING TEST DATA FROM FILE -----
	auto testingData = dataParser.readData("../../data/hats/test2.txt");
	
	//  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(5000);
  	qtDisplay.trackLayer(1);
  	std::cout << network.getNeurons().back().getNeuronID() << std::endl;
	qtDisplay.trackNeuron(network.getNeurons().back().getNeuronID());
	
    //  ----- RUNNING THE NETWORK -----
    network.run(&trainingData, &testingData, 0.1);
	analysis.accuracy();
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
