/*
 * unsupervisedNetwork.cpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 29/10/2018
 *
 * Information: Example of a spiking neural network that can learn one dimensional patterns.
 */

#include <iostream>

#include "../source/neuron.hpp"
#include "../source/dataParser.hpp"
#include "../source/network.hpp"
#include "../source/qtDisplay.hpp"
#include "../source/spikeLogger.hpp"
#include "../source/stdp.hpp"
#include "../source/myelinPlasticityLogger.hpp"
#include "../source/myelinPlasticity.hpp"

int main(int argc, char** argv)
{
    //  ----- READING TRAINING DATA FROM FILE -----
	adonis::DataParser dataParser;
	
	auto trainingData = dataParser.readData("../../data/1D_patterns/oneD_10neurons_4patterns_.txt");
	
    //  ----- INITIALISING THE NETWORK -----
	adonis::QtDisplay qtDisplay;
	adonis::SpikeLogger spikeLogger("10neurons_4patterns_unsupervised_spikeLog.bin");
	adonis::MyelinPlasticityLogger myelinPlasticityLogger("10neurons_4patterns_unsupervised_learningLog.bin");
	adonis::Network network({&spikeLogger, &myelinPlasticityLogger}, &qtDisplay);

    //  ----- NETWORK PARAMETERS -----
	float decayCurrent = 10;
	float potentialDecay = 20;
	float refractoryPeriod = 3;

    int inputNeurons = 10;
    int layer1Neurons = 4;
	
	float eligibilityDecay = 20;
    float weight = 1./10;
	
	bool wta = true;
	bool burst = false;
	bool homeostasis = false;
	
	//  ----- INITIALISING THE LEARNING RULE -----
	adonis::MyelinPlasticity myelinPlasticity;
	adonis::STDP stdp;
	
    //  ----- CREATING THE NETWORK -----
	network.addLayer({}, inputNeurons, 1, 1, false, decayCurrent, potentialDecay, refractoryPeriod, wta, false, eligibilityDecay);
	network.addLayer({&myelinPlasticity}, layer1Neurons, 1, 1, homeostasis, decayCurrent, potentialDecay, 100, wta, burst, eligibilityDecay);
	
	//  ----- CONNECTING THE NETWORK -----
	network.allToAll(network.getLayers()[0], network.getLayers()[1], weight, 0, 5, 3);
	
    //  ----- DISPLAY SETTINGS -----
	qtDisplay.useHardwareAcceleration(true);
	qtDisplay.setTimeWindow(5000);
	qtDisplay.trackNeuron(11);

	network.turnOffLearning(80000);

    //  ----- RUNNING THE NETWORK -----
    network.run(0.1, &trainingData);
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
