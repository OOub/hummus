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
	float eligibilityDecay = 20;
	
    //  ----- CREATING THE NETWORK -----
    adonis::MyelinPlasticity mp(1, 1, 1, 1);
    adonis::STDP stdp(1, 1, 20, 20);
    
    network.add2dLayer<adonis::InputNeuron>(0, 4, 28, 28, 1, true, {});
    network.add2dLayer<adonis::LIF>(0, 4, 28, 28, 1, true, {&mp}, 900, true, 10, 20, eligibilityDecay);
    network.add2dLayer<adonis::LIF>(0, 4, 14, 14, 1, true, {}, 900, true, 10, 20, eligibilityDecay);
    network.addDecisionMakingLayer<adonis::DecisionMakingNeuron>("../../data/cards/trainHatsLabel.txt", {&mp}, 900, true, 10, 80, 80);
    
    //  ----- CONNECTING THE NETWORK -----
    network.convolution(network.getLayers()[0], network.getLayers()[1], 0.03, 0.02, 5, 3);
    network.pooling(network.getLayers()[0], network.getLayers()[1], 1);
    network.allToAll(network.getLayers()[1], network.getLayers()[2], 0.3, 0.2, 5, 3);
	
	//  ----- READING DATA FROM FILE -----
    adonis::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/pokerDVS/trainHats.txt");
    auto testData = dataParser.readData("../../data/pokerDVS/testHats.txt");
	
	//  ----- DISPLAY SETTINGS -----
	qtDisplay.useHardwareAcceleration(true);
	qtDisplay.setTimeWindow(20000);
	qtDisplay.trackLayer(1);
    qtDisplay.trackNeuron(network.getNeurons().back()->getNeuronID());
	
    //  ----- RUNNING THE NETWORK -----
    network.run(&trainingData, 0.1, &testData);
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
