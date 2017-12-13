/*
 * smallNetwork.cpp
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
	
	// clean signal test
//  	auto data = dataParser.read1D("../../data/generatedPatterns/cleanSignal/0bn0nn4fakePatterns_snnTest_2000reps_10msInterval.txt");
	
	// time jitter test
	auto data = dataParser.read1D("../../data/generatedPatterns/timeJitter/1.5timeJitter0bn0nn4fakePatterns_snnTest_2000reps_10msInterval.txt");
	
    // additive noise test
//	auto data = dataParser.read1D("../../data/generatedPatterns/additiveNoise/10bn0nn4fakePatterns_snnTest_2000reps_10msInterval.txt");
	
//  ----- NETWORK PARAMETERS -----
	
//	std::string filename = "ThresholdDecelerate.bin";
//	baal::Logger logger(filename);
	baal::Display network;
	
//  ----- INITIALISING THE NETWORK -----
	float runtime = data[0].back()+1;
	float timestep = 0.1;
	
	float decayCurrent = 3;
	float potentialDecay = 20;
	float refractoryPeriod = 3;
    float efficacyDecay = 500;
    float efficacy = 1;
	
    int inputNeurons = 27;
    int layer1Neurons = 27;
	
    float weight = 1./4;
	
	network.addNeurons(inputNeurons, decayCurrent, potentialDecay, refractoryPeriod, efficacyDecay, efficacy);
	network.addNeurons(layer1Neurons, decayCurrent, potentialDecay, refractoryPeriod, efficacyDecay, efficacy);
	
	network.allToallConnectivity(&network.getNeuronPopulations()[0], &network.getNeuronPopulations()[1], weight, true, 20);

	// injecting spikes in the input layer
	for (auto idx=0; idx<data[0].size(); idx++)
	{
		network.injectSpike(network.getNeuronPopulations()[0][data[1][idx]].prepareInitialSpike(data[0][idx]));
    }
	
//  ----- DISPLAY SETTINGS -----
	network.useHardwareAcceleration(true);
	network.setTimeWindow(1000);
	network.setOutputMinY(layer1Neurons);
	network.trackNeuron(32);
	
//  ----- RUNNING THE NETWORK -----
    int errorCode = network.run(runtime, timestep);
	
//  ----- EXITING APPLICATION -----
    return errorCode;
}
