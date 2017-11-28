/*
 * smallNetwork.cpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 17/11/2017
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
    float efficacyDecay = 1000;
    float efficacy = 1;
	float threshold=-50;
	float restingPotential=-70;
	float resetPotential=-70;
	float inputResistance=50e9;
	float externalCurrent=90e-10;
	float currentBurnout=3.1e-9;
	
    int inputNeurons = 100;
    int layer1Neurons = 100;
	
    float weight = 0.025;
	
	network.addNeurons(inputNeurons, decayCurrent, potentialDecay, refractoryPeriod, efficacyDecay, efficacy, threshold, restingPotential, resetPotential, inputResistance, externalCurrent, currentBurnout);
	network.addNeurons(layer1Neurons, decayCurrent, potentialDecay, refractoryPeriod, efficacyDecay, efficacy, threshold, restingPotential, resetPotential, inputResistance, externalCurrent, currentBurnout);
	
	network.allToallConnectivity(&network.getNeuronPopulations()[0], &network.getNeuronPopulations()[1], weight, true, 20);
	
	network.injectSpike(network.getNeuronPopulations()[0][0].prepareInitialSpike(5));
	network.injectSpike(network.getNeuronPopulations()[0][1].prepareInitialSpike(8));
	network.injectSpike(network.getNeuronPopulations()[0][8].prepareInitialSpike(10));
	network.injectSpike(network.getNeuronPopulations()[0][8].prepareInitialSpike(20));
	
//  ----- DISPLAY SETTINGS -----
	network.useHardwareAcceleration(true);
	network.setTimeWindow(runtime);
	network.setOutputMinY(layer1Neurons);
	network.trackNeuron(10);
	
//  ----- RUNNING THE NETWORK -----
    int errorCode = network.run(runtime, timestep);
	
//  ----- EXITING APPLICATION -----
    return errorCode;
}
