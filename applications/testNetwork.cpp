/*
 * testNetwork.cpp
 * Adonis_t - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/05/2018
 *
 * Information: Example of a basic spiking neural network.
 */

#include <iostream>

#include "../source/network.hpp"
#include "../source/display.hpp"
#include "../source/spikeLogger.hpp"

int main(int argc, char** argv)
{
	adonis_t::DataParser dataParser;

//  ----- NETWORK PARAMETERS -----
	adonis_t::Display network;

//  ----- INITIALISING THE NETWORK -----
	float runtime = 100;
	float timestep = 0.1;

	float decayCurrent = 10;
	float potentialDecay = 20;
	float refractoryPeriod = 3;

    int inputNeurons = 1;
    int layer1Neurons = 1;
	int layer2Neurons = 1;

    float weight = 19e-10;

	// creating input neurons
	network.addNeurons(0, adonis_t::learningMode::noLearning, inputNeurons, decayCurrent, potentialDecay, refractoryPeriod);

	// creating layer 1 neurons
	network.addNeurons(1, adonis_t::learningMode::noLearning, layer1Neurons, decayCurrent, potentialDecay, refractoryPeriod);

	// creating layer 2 neurons
	network.addNeurons(2, adonis_t::learningMode::noLearning, layer2Neurons, decayCurrent, potentialDecay, refractoryPeriod);

	// connecting input layer and layer 1
	network.allToallConnectivity(&network.getNeuronPopulations()[0].rfNeurons, &network.getNeuronPopulations()[1].rfNeurons, false, weight, false, 0);

	// connecting layer 1 and layer 2
	network.allToallConnectivity(&network.getNeuronPopulations()[1].rfNeurons, &network.getNeuronPopulations()[2].rfNeurons, false, weight, false, 0);

	// injecting spikes in the input layer
	network.injectSpike(network.getNeuronPopulations()[0].rfNeurons[0].prepareInitialSpike(10));
	network.injectSpike(network.getNeuronPopulations()[0].rfNeurons[0].prepareInitialSpike(11));
	network.injectSpike(network.getNeuronPopulations()[0].rfNeurons[0].prepareInitialSpike(13));
	network.injectSpike(network.getNeuronPopulations()[0].rfNeurons[0].prepareInitialSpike(15));
	network.injectSpike(network.getNeuronPopulations()[0].rfNeurons[0].prepareInitialSpike(25));

//  ----- DISPLAY SETTINGS -----
	network.useHardwareAcceleration(true);
	network.setTimeWindow(runtime);
	network.trackNeuron(2);

//  ----- RUNNING THE NETWORK -----
    int errorCode = network.run(runtime, timestep);

//  ----- EXITING APPLICATION -----
    return errorCode;
}
