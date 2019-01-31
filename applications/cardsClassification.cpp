/*
 * cardsClassification.cpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
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
    adonis::SpikeLogger spikeLog("spikeLog.bin");
    adonis::PredictionLogger predictionLog("predictionLog.bin");
    adonis::MyelinPlasticityLogger mpLog("mpLog.bin");
    
    adonis::Network network({&spikeLog, &predictionLog, &mpLog}, &qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	float eligibilityDecay = 100;
	
    //  ----- CREATING THE NETWORK -----
    adonis::MyelinPlasticity mp(1, 1, 1, 1);
    
    network.add2dLayer<adonis::InputNeuron>(0, 1, 34, 34, 1, false, {});
    network.addDecisionMakingLayer<adonis::DecisionMakingNeuron>("../../data/cards/heart1trainLabel.txt", {&mp}, 1000, true, 10, 20, eligibilityDecay, 0);
    
    //  ----- CONNECTING THE NETWORK -----
    network.allToAll(network.getLayers()[0], network.getLayers()[1], 0.006, 0.02, 10, 5);
	
	//  ----- READING DATA FROM FILE -----
    adonis::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/cards/heart1train.txt");
    auto testData = dataParser.readData("../../data/cards/heart9test.txt");
	
	//  ----- DISPLAY SETTINGS -----
	qtDisplay.useHardwareAcceleration(true);
	qtDisplay.setTimeWindow(5000);
	qtDisplay.trackLayer(1);
    qtDisplay.trackNeuron(network.getNeurons().back()->getNeuronID());
	
    //  ----- RUNNING THE NETWORK -----
    network.run(&trainingData, 1, &testData);
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
