/*
 * pokerDVS_HATS.cpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/01/2019
 *
 * Information: Spiking neural network classifying the poker-DVS dataset
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/GUI/qtDisplay.hpp"
#include "../source/learningRules/stdp.hpp"
#include "../source/learningRules/myelinPlasticity.hpp"
#include "../source/neurons/inputNeuron.hpp"
#include "../source/neurons/decisionMakingNeuron.hpp"
#include "../source/neurons/leakyIntegrateAndFire.hpp"
#include "../source/addOns/spikeLogger.hpp"
#include "../source/addOns/predictionLogger.hpp"
#include "../source/addOns/myelinPlasticityLogger.hpp"

int main(int argc, char** argv) {
    //  ----- INITIALISING THE NETWORK -----
    adonis::QtDisplay qtDisplay;
    
    adonis::Network network(&qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	float eligibilityDecay = 100;
	
    //  ----- CREATING THE NETWORK -----
    adonis::MyelinPlasticity mp(1, 1, 1, 1);
    
    network.add2dLayer<adonis::InputNeuron>(0, 1, 28, 28, 1, false, {});
    network.addLayer<adonis::LIF>(100, 1, 1, {&mp}, true, 10, 20, 3, true, false, eligibilityDecay);
    network.addDecisionMakingLayer<adonis::DecisionMakingNeuron>("../../data/pokerDVS/trainHatsLabel.txt", false, {&mp}, 900, false, 10, 20, eligibilityDecay);
    
    //  ----- CONNECTING THE NETWORK -----
    network.allToAll(network.getLayers()[0], network.getLayers()[1], 0.04, 0.02, 5, 3);
    network.allToAll(network.getLayers()[1], network.getLayers()[2], 0.4, 0.2, 5, 3);
	
	//  ----- READING DATA FROM FILE -----
    adonis::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/pokerDVS/trainHats.txt");
    auto testData = dataParser.readData("../../data/pokerDVS/testHats.txt");
	
	//  ----- DISPLAY SETTINGS -----
	qtDisplay.useHardwareAcceleration(true);
	qtDisplay.setTimeWindow(10000);
    qtDisplay.trackLayer(1);
    qtDisplay.trackNeuron(network.getNeurons().back()->getNeuronID());
	
    //  ----- RUNNING THE NETWORK -----
    network.run(&trainingData, 1, &testData);
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
