/*
 * unsupervisedNetwork.cpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 6/12/2017
 *
 * Information: Example of a basic spiking neural network.
 */

#include <iostream>

#include "../source/neuron.hpp"
#include "../source/dataParser.hpp"
#include "../source/network.hpp"
#include "../source/qtDisplay.hpp"
#include "../source/spikeLogger.hpp"
#include "../source/learningLogger.hpp"

int main(int argc, char** argv)
{
    //  ----- READING TRAINING DATA FROM FILE -----
	adonis_c::DataParser dataParser;
	
	auto trainingData = dataParser.readTrainingData("../../data/1D_patterns/timeJitter/oneD_10neurons_4patterns.txt");
	
    //  ----- INITIALISING THE NETWORK -----
	adonis_c::QtDisplay qtDisplay;
	adonis_c::SpikeLogger spikeLogger("10neurons_4patterns_unsupervised_spikeLog.bin");
	adonis_c::LearningLogger learningLogger("10neurons_4patterns_unsupervised_learningLog.bin");
	adonis_c::Network network({&spikeLogger, &learningLogger}, &qtDisplay);

    //  ----- NETWORK PARAMETERS -----
	float runtime = trainingData.back().timestamp+1;

	float timestep = 0.1;

	float decayCurrent = 10;
	float potentialDecay = 20;
	
	float refractoryPeriod = 3;

    int inputNeurons = 10;
    int layer1Neurons = 4;
	
	float alpha = 1;
	float lambda = 0.1;
	float eligibilityDecay = 20;
    float weight = 19e-10/10; //weight dependent on feature size

    //  ----- CREATING THE NETWORK -----
	network.addNeurons(0, adonis_c::learningMode::noLearning, inputNeurons,decayCurrent, potentialDecay, refractoryPeriod, eligibilityDecay, alpha, lambda);
	network.addNeurons(1, adonis_c::learningMode::myelinPlasticityNoReinforcement, layer1Neurons, decayCurrent, potentialDecay, refractoryPeriod, eligibilityDecay, alpha, lambda);
	
	//  ----- CONNECTING THE NETWORK -----
	network.allToAllConnectivity(&network.getNeuronPopulations()[0].rfNeurons, &network.getNeuronPopulations()[1].rfNeurons, false, weight, true, 10);
	
	//  ----- INJECTING SPIKES -----
	for (auto idx=0; idx<trainingData.size(); idx++)
	{
		network.injectSpike(network.getNeuronPopulations()[0].rfNeurons[trainingData[idx].neuronID].prepareInitialSpike(trainingData[idx].timestamp));
    }

    //  ----- DISPLAY SETTINGS -----
	qtDisplay.useHardwareAcceleration(true);
	qtDisplay.setTimeWindow(1000);
	qtDisplay.trackNeuron(11);

	network.turnOffLearning(80000);

    //  ----- RUNNING THE NETWORK -----
    network.run(runtime, timestep);
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
