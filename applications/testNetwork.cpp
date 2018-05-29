/*
 * testNetwork.cpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 12/01/2018
 *
 * Information: Example of a basic spiking neural network.
 */

#include <iostream>

#include "../source/network.hpp"
#include "../source/display.hpp"
#include "../source/logger.hpp"

int main(int argc, char** argv)
{
	baal::DataParser dataParser;

//  ----- NETWORK PARAMETERS -----
	baal::Display network;

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
	network.addNeurons(inputNeurons, 0, baal::learningMode::noLearning, decayCurrent, potentialDecay, refractoryPeriod);

	// creating layer 1 neurons
	network.addNeurons(layer1Neurons, 1, baal::learningMode::noLearning, decayCurrent, potentialDecay, refractoryPeriod);

	// creating layer 2 neurons
	network.addNeurons(layer2Neurons, 2, baal::learningMode::noLearning, decayCurrent, potentialDecay, refractoryPeriod);

	// connecting input layer and layer 1
	network.allToallConnectivity(&network.getNeuronPopulations()[0].rfNeurons, &network.getNeuronPopulations()[1].rfNeurons, false, weight, false, 0); // the bool allows us to randomize weights and delays

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
