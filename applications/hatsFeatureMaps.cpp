/*
 * hatsNetwork.cpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: Spiking neural network running with histograms of averaged time surfaces converted into spikes.
 */

#include <iostream> 

#include "../source/core.hpp"
#include "../source/dataParser.hpp"
#include "../source/addOns/analysis.hpp"
#include "../source/GUI/qtDisplay.hpp"
#include "../source/addOns/predictionLogger.hpp"
#include "../source/learningRules/rewardModulatedSTDP.hpp"
#include "../source/learningRules/stdp.hpp"
#include "../source/neurons/inputNeuron.hpp"
#include "../source/neurons/decisionMakingNeuron.hpp"
#include "../source/neurons/leakyIntegrateAndFire.hpp"

int main(int argc, char** argv) {
    //  ----- INITIALISING THE NETWORK -----
	adonis::QtDisplay qtDisplay;
	adonis::Analysis analysis("../../data/hats/testLabel.txt");
	adonis::Network network({&analysis}, &qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	float decayCurrent = 10;
	float decayPotential = 20;
	float refractoryPeriod = 3;
	float eligibilityDecay = 20;
	
	bool burst = false;
	bool wta = false;
	bool homeostasis = false;
	
	//  ----- INITIALISING THE LEARNING RULE -----
	adonis::STDP stdp;
	adonis::RewardModulatedSTDP rstdp;
	
	//  ----- CREATING THE NETWORK -----
    network.addLayer<adonis::InputNeuron>(1470, 1, 6, {});
    network.addLayer<adonis::LIF>(10, 1, 6, {&stdp}, homeostasis, decayCurrent, decayPotential+20, refractoryPeriod, wta, burst, eligibilityDecay+20);
	network.addDecisionMakingLayer<adonis::DecisionMakingNeuron>("../../data/hats/trainLabel.txt", {&rstdp}, 100);
	
    network.allToAll(network.getLayers()[0], network.getLayers()[1], 0.0006, 0.0004, 2, 0, 100);
    network.allToAll(network.getLayers()[1], network.getLayers()[2], 0.6, 0.4, 5, 3);
    network.lateralInhibition(network.getLayers()[1], -1);
	
    //  ----- READING TRAINING DATA FROM FILE -----
    adonis::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/hats/train.txt");
    
    //  ----- READING TEST DATA FROM FILE -----
    auto testData = dataParser.readData("../../data/hats/test.txt");
	
	//  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(5000);
  	qtDisplay.trackLayer(2);
    qtDisplay.trackNeuron(network.getNeurons().back()->getNeuronID());
	
    //  ----- RUNNING THE NETWORK -----
    network.run(&trainingData, 0.1, &testData);
	analysis.accuracy();
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
