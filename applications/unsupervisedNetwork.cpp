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
	auto data = dataParser.read1D("../../data/generatedPatterns/timeJitter/3timeJitter0bn0nn4fakePatterns_snnTest_2000reps_10msInterval.txt");
	
//  ----- NETWORK PARAMETERS -----
	std::string filename = "unsupervisedLearning_jitter.bin";

	baal::Logger logger(filename);
	baal::Display network({&logger});

//  ----- INITIALISING THE NETWORK -----
	float runtime = data.back().timestamp+1;

	float timestep = 0.1;

	float decayCurrent = 10;
	float potentialDecay = 20;

	float decayCurrent2 = 40;
	float potentialDecay2 = 50;
	
	float refractoryPeriod = 3;

    int inputNeurons = 27;
    int layer1Neurons = 27;
	int layer2Neurons = 27;
	
	float alpha = 0.25;
	float lambda = 0.25;
	float eligibilityDecay = 100;

    float weight = 19e-10/4; //weight dependent on feature size

	network.addNeurons(inputNeurons, decayCurrent, potentialDecay, refractoryPeriod, eligibilityDecay, alpha, lambda);
	network.addNeurons(layer1Neurons, decayCurrent, potentialDecay, refractoryPeriod, eligibilityDecay, alpha, lambda);
	network.addNeurons(layer2Neurons, decayCurrent2, potentialDecay2, refractoryPeriod, eligibilityDecay, alpha, lambda);
	
	network.allToallConnectivity(&network.getNeuronPopulations()[0], &network.getNeuronPopulations()[1], false, weight, true, 20);
	network.allToallConnectivity(&network.getNeuronPopulations()[1], &network.getNeuronPopulations()[2], false, weight, true, 30);
	
	// injecting spikes in the input layer
	for (auto idx=0; idx<data.size(); idx++)
	{
		network.injectSpike(network.getNeuronPopulations()[0][data[idx].neuronID].prepareInitialSpike(data[idx].timestamp));
    }

//  ----- DISPLAY SETTINGS -----
	network.useHardwareAcceleration(true);
	network.setTimeWindow(1000);
	network.setOutputMinY(layer1Neurons);
	network.trackNeuron(28);

//  ----- RUNNING THE NETWORK -----
    int errorCode = network.run(runtime, timestep);

//  ----- EXITING APPLICATION -----
    return errorCode;
}
