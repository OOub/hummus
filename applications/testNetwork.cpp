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
	
//  ----- NETWORK PARAMETERS -----
	baal::Display network;
	
//  ----- INITIALISING THE NETWORK -----
	float runtime = 100;
	float timestep = 0.1;
	
	float decayCurrent = 10;
	float potentialDecay = 20;
	float refractoryPeriod = 3;
    float efficacyDecay = 0;
    float efficacy = 1;
	
    int inputNeurons = 1;
    int layer1Neurons = 1;
	
    float weight = 19e-10/2;
	
	network.addNeurons(inputNeurons, decayCurrent, potentialDecay, refractoryPeriod, efficacyDecay, efficacy);
	network.addNeurons(layer1Neurons, decayCurrent, potentialDecay, refractoryPeriod, efficacyDecay, efficacy);
	
	network.allToallConnectivity(&network.getNeuronPopulations()[0], &network.getNeuronPopulations()[1], false, weight, false, 0); // the bool refers to whether or not we want to randomize the weights and delays
	
	// injecting spikes in the input layer
	network.injectSpike(network.getNeuronPopulations()[0][0].prepareInitialSpike(10));
	network.injectSpike(network.getNeuronPopulations()[0][0].prepareInitialSpike(11));
	network.injectSpike(network.getNeuronPopulations()[0][0].prepareInitialSpike(13));
	network.injectSpike(network.getNeuronPopulations()[0][0].prepareInitialSpike(15));
	network.injectSpike(network.getNeuronPopulations()[0][0].prepareInitialSpike(25));
	
//  ----- DISPLAY SETTINGS -----
	network.useHardwareAcceleration(true);
	network.setTimeWindow(runtime);
	network.setOutputMinY(layer1Neurons);
	network.trackNeuron(1);
	
//  ----- RUNNING THE NETWORK -----
    int errorCode = network.run(runtime, timestep);
	
//  ----- EXITING APPLICATION -----
    return errorCode;
}
