/*
 * hatsNetwork.cpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: spiking neural network running the n-Cars database with HATS encoded with the Intentisty-to-latency method;
 */

#include <iostream> 

#include "../source/GUI/qtDisplay.hpp"
#include "../source/core.hpp"
#include "../source/dataParser.hpp"
#include "../source/addOns/analysis.hpp"
#include "../source/learningRules/myelinPlasticity.hpp"
#include "../source/neurons/inputNeuron.hpp"
#include "../source/neurons/decisionMakingNeuron.hpp"

int main(int argc, char** argv)
{
    //  ----- INITIALISING THE NETWORK -----
    adonis::QtDisplay qtDisplay;
	adonis::Analysis analysis("../../data/hats/testLabel2.txt");
	adonis::Network network({&analysis}, &qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	float decayCurrent = 10;
	float decayPotential = 20;
	float eligibilityDecay = 100;
	
	//  ----- INITIALISING THE LEARNING RULE -----
	adonis::MyelinPlasticity mp(0.1, 0.1, true);
	
	//  ----- CREATING THE NETWORK -----
    network.addLayer<adonis::InputNeuron>(1470, 1, 1, {});
    network.addDecisionMakingLayer<adonis::DecisionMakingNeuron>("../../data/hats/trainLabel2.txt", {&mp}, 900, false, decayCurrent, decayPotential, eligibilityDecay+50);
	
	network.allToAll(network.getLayers()[0], network.getLayers()[1], 0.05, 0.02, 5, 3, 100);
	
	//  ----- READING TRAINING DATA FROM FILE -----
	adonis::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/hats/train2.txt");
	
	//  ----- READING TEST DATA FROM FILE -----
	auto testingData = dataParser.readData("../../data/hats/train2.txt");
	
	//  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(5000);
  	qtDisplay.trackLayer(2);
	 qtDisplay.trackNeuron(network.getNeurons().back()->getNeuronID());
	
    //  ----- RUNNING THE NETWORK -----
    network.run(&trainingData, 1 , &testingData);
	analysis.accuracy();
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
