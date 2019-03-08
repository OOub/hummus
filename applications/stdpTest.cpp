/*
 * stdpPotentiation.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: Example of stdp working. 10 neurons are connected to an output neuron. In the beginning, all 10 neurons are needed to fire (disable the learning rule to see that). With
 * STDP, postsynaptic firing slowly shifts and the neurons that fire after the output neuron get depressed (use the debug option to see the weight progression as the network is
 * running)
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/randomDistributions/normal.hpp"
#include "../source/GUI/qtDisplay.hpp"
#include "../source/learningRules/stdp.hpp"
#include "../source/learningRules/timeInvariantSTDP.hpp"
#include "../source/learningRules/myelinPlasticity.hpp"
#include "../source/learningRules/rewardModulatedSTDP.hpp"
#include "../source/neurons/input.hpp"
#include "../source/neurons/LIF.hpp"
#include "../source/neurons/decisionMaking.hpp"

int main(int argc, char** argv) {
    //  ----- READING TRAINING DATA FROM FILE -----
	hummus::DataParser dataParser;
	
	auto trainingData = dataParser.readData("../../data/stdpTest.txt");
	
    //  ----- INITIALISING THE NETWORK -----
	hummus::QtDisplay qtDisplay;
	hummus::Network network(&qtDisplay);

    //  ----- NETWORK PARAMETERS -----
	float resetCurrent = 10;
	float potentialDecay = 20;
	float refractoryPeriod = 30;
	
    int inputNeurons = 10;
    int layer1Neurons = 1;
	
    float weight = 1./10;
	
	//  ----- INITIALISING THE LEARNING RULE -----
    auto stdp = network.makeLearningRule<hummus::STDP>();
	
	//  ----- CREATING THE NETWORK -----
    network.addLayer<hummus::Input>(inputNeurons, {});
    network.addLayer<hummus::LIF>(layer1Neurons, {stdp}, true, false, resetCurrent, potentialDecay, refractoryPeriod);

    //  ----- CONNECTING THE NETWORK -----
    network.allToAll(network.getLayers()[0], network.getLayers()[1], hummus::Normal(weight, 0, 1, 0));
	
    //  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(100);
  	qtDisplay.trackNeuron(10);
  	qtDisplay.trackLayer(1);
	
    //  ----- RUNNING THE NETWORK -----
    network.run(&trainingData, 0.1);

    //  ----- EXITING APPLICATION -----
    return 0;
}
