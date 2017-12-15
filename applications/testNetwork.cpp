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
	std::string filename = "test.bin";
	baal::Logger logger(filename);
	baal::Display network({&logger});
	
//  ----- INITIALISING THE NETWORK -----
	float runtime = 50;
	float timestep = 0.1;
	
	float decayCurrent = 10;
	float potentialDecay = 20;
	float refractoryPeriod = 3;
    float efficacyDecay = 0;
    float efficacy = 1;
	
    int inputNeurons = 4;
    int layer1Neurons = 2;
	
    float weight = 19e-10/4;
	
	network.addNeurons(inputNeurons, decayCurrent, potentialDecay, refractoryPeriod, efficacyDecay, efficacy);
	network.addNeurons(layer1Neurons, decayCurrent, potentialDecay, refractoryPeriod, efficacyDecay, efficacy);
	
	network.allToallConnectivity(&network.getNeuronPopulations()[0], &network.getNeuronPopulations()[1], weight, false, 0);

	network.injectSpike(network.getNeuronPopulations()[0][0].prepareInitialSpike(5));
	network.injectSpike(network.getNeuronPopulations()[0][1].prepareInitialSpike(5));
	network.injectSpike(network.getNeuronPopulations()[0][2].prepareInitialSpike(5));
	network.injectSpike(network.getNeuronPopulations()[0][3].prepareInitialSpike(5));
	
//  ----- DISPLAY SETTINGS -----
	network.useHardwareAcceleration(true);
	network.setTimeWindow(runtime);
	network.setOutputMinY(layer1Neurons);
	network.trackNeuron(4);
	
//  ----- RUNNING THE NETWORK -----
    int errorCode = network.run(runtime, timestep);
	
//  ----- EXITING APPLICATION -----
    return errorCode;
}
