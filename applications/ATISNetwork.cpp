/*
 * ATISNetwork.cpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 27/02/2018
 *
 * Information: Example of a basic spiking neural network.
 */

#include <iostream>

#include "../source/dataParser.hpp"
#include "../source/network.hpp"
#include "../source/display.hpp"
#include "../source/logger.hpp"

int main(int argc, char** argv)
{
//  ----- READING DATA FROM FILE -----
	baal::DataParser dataParser;
	
	auto data = dataParser.readData("../../data/pip/1rec_1pip/1pip_1type_200reps.txt");
	
//  ----- NETWORK PARAMETERS -----
	std::string filename = "unsupervised_ATIS1pip.bin";
	baal::Logger logger(filename);
	baal::Display network({&logger});

//  ----- INITIALISING THE NETWORK -----
	float runtime = data.back().timestamp+1;
	float timestep = 0.1;
	
	float decayCurrent = 20;
	float potentialDecay = 30;
	
	float decayCurrent2 = 300;
	float potentialDecay2 = 310;
	
	float refractoryPeriod = 3;
	
    int inputNeurons = 671;
    int layer1Neurons = 10;
	int layer2Neurons = 10;
	
	float alpha = 1;
	float lambda = 1;
	
	float eligibilityDecay = 100;
	
	network.addNeurons(inputNeurons, decayCurrent, potentialDecay, refractoryPeriod, eligibilityDecay, alpha, lambda);
	network.addNeurons(layer1Neurons, decayCurrent, potentialDecay, refractoryPeriod, eligibilityDecay, alpha, lambda);
	network.addNeurons(layer2Neurons, decayCurrent2, potentialDecay2, refractoryPeriod, 300, alpha, lambda);
	
	network.allToallConnectivity(&network.getNeuronPopulations()[0], &network.getNeuronPopulations()[1], false, 50e-10/10, true, 100);
	network.allToallConnectivity(&network.getNeuronPopulations()[1], &network.getNeuronPopulations()[2], false, 50e-10, true, 300);
	
	// injecting spikes in the input layer
	for (auto idx=0; idx<data.size(); idx++)
	{
		network.injectSpike(network.getNeuronPopulations()[0][data[idx].neuronID].prepareInitialSpike(data[idx].timestamp));
    }

//  ----- DISPLAY SETTINGS -----
	network.useHardwareAcceleration(true);
	network.setTimeWindow(10000);
	network.trackNeuron(780);
	network.trackLayer(2);
	
//  ----- RUNNING THE NETWORK -----
    int errorCode = network.run(runtime, timestep);

//  ----- EXITING APPLICATION -----
    return errorCode;
}
