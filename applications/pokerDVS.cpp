/*
 * pokerDVS.cpp
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
#include "../source/neurons/LIF.hpp"
#include "../source/addOns/spikeLogger.hpp"
#include "../source/addOns/classificationLogger.hpp"
#include "../source/addOns/myelinPlasticityLogger.hpp"

int main(int argc, char** argv) {
    //  ----- INITIALISING THE NETWORK -----
    adonis::QtDisplay qtDisplay;
    adonis::SpikeLogger spikeLog("spikeLog.bin");
    adonis::ClassificationLogger classificationLog("predictionLog.bin");
    adonis::MyelinPlasticityLogger mpLog("mpLog.bin");
    
    adonis::Network network({&spikeLog, &classificationLog, &mpLog}, &qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	float eligibilityDecay = 100;
	
    //  ----- CREATING THE NETWORK -----
    adonis::MyelinPlasticity mp(1, 1, 1, 1);
    
    network.add2dLayer<adonis::InputNeuron>(0, 1, 34, 34, 1, false, {});
    network.add2dLayer<adonis::LIF>(0, 1, 34, 34, 1, false, {&mp}, true, true, 10, 20, 3, true, false, eligibilityDecay);
    network.addDecisionMakingLayer<adonis::DecisionMakingNeuron>("../../data/pokerDVS/DHtrainLabel.txt", true, {&mp}, true, false, 10, 80, 1000, eligibilityDecay);
    
    //  ----- CONNECTING THE NETWORK -----
    network.allToAll(network.getLayers()[0], network.getLayers()[1], 0.006, 0.02, 50, 10);
    network.allToAll(network.getLayers()[1], network.getLayers()[2], 0.06, 0.02);
	
	//  ----- READING DATA FROM FILE -----
    adonis::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/pokerDVS/DHtrain.txt");
    auto testData = dataParser.readData("../../data/pokerDVS/DHtest.txt");
	
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
