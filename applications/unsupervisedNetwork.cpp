/*
 * unsupervisedNetwork.cpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 6/12/2017
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
	
	// time jitter test
	auto data = dataParser.read1D("../../data/generatedPatterns/timeJitter/3timeJitter0bn0nn4fakePatterns_snnTest_400reps_10msInterval.txt");
	
//  ----- NETWORK PARAMETERS -----
	
	std::string filename = "unsupervisedLearning_jitter.bin";
	
	baal::Logger logger(filename);
	baal::Display network({&logger});
	
//  ----- INITIALISING THE NETWORK -----
	float runtime = data[0].back()+1;
	float timestep = 0.1;
	
	float decayCurrent = 10;
	float potentialDecay = 20;
	
	float decayCurrent2 = 50;
	float potentialDecay2 = 60;
	
	float refractoryPeriod = 3;
	
    int inputNeurons = 27;
    int layer1Neurons = 27;
	int layer2Neurons = 10;

	float alpha = 0.1;
	float lambda = 2;
	
    float weight = 19e-10/4; //weight dependent on feature size
	
	network.addNeurons(inputNeurons, decayCurrent, potentialDecay, refractoryPeriod, alpha, lambda);
	network.addNeurons(layer1Neurons, decayCurrent, potentialDecay, refractoryPeriod, alpha, lambda);
	network.addNeurons(layer2Neurons, decayCurrent2, potentialDecay2, refractoryPeriod, alpha, lambda);
	
	network.allToallConnectivity(&network.getNeuronPopulations()[0], &network.getNeuronPopulations()[1], false, weight, true, 20);
	network.allToallConnectivity(&network.getNeuronPopulations()[1], &network.getNeuronPopulations()[2], false, 0, false, 0); // weight has to be different
	
	// injecting spikes in the input layer
	for (auto idx=0; idx<data[0].size(); idx++)
	{
		network.injectSpike(network.getNeuronPopulations()[0][data[1][idx]].prepareInitialSpike(data[0][idx]));
    }
	
//  ----- DISPLAY SETTINGS -----
	network.useHardwareAcceleration(true);
	network.setTimeWindow(1000);
	network.setOutputMinY(layer1Neurons);
	network.trackNeuron(37);
	
//  ----- RUNNING THE NETWORK -----
    int errorCode = network.run(runtime, timestep);
	
//  ----- EXITING APPLICATION -----
    return errorCode;
}
