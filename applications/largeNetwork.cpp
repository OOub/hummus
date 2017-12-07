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
  	auto data = dataParser.read1D("../../data/generatedPatterns/cleanSignal/0bn0nn4fakePatterns_snnTest_2000reps_10msInterval.txt");
	
	// time jitter test
//	auto data = dataParser.read1D("../../data/generatedPatterns/timeJitter/1.5timeJitter0bn0nn4fakePatterns_snnTest_2000reps_10msInterval.txt");

    // additive noise test
//	auto data = dataParser.read1D("../../data/generatedPatterns/additiveNoise/10bn0nn4fakePatterns_snnTest_2000reps_10msInterval.txt");
	
	// supervised learning test
//	auto data = dataParser.read1D("../../data/thresholdAdaptationTest.txt");
//	auto teacher = dataParser.read1D("../../data/teacherSignalDecelerate.txt");
	
//  ----- NETWORK PARAMETERS -----
	
	std::string filename = "ThresholdDecelerate.bin";
	baal::Logger logger(filename);
	baal::Display network({&logger});
	
//  ----- INITIALISING THE NETWORK -----
	float runtime = data[0].back()+1;
	float timestep = 0.1;
	
	float decayCurrent = 3;
	float potentialDecay = 20;
	float refractoryPeriod = 3;
    float efficacyDecay = 1000;
    float efficacy = 1;
	
    int inputNeurons = 27;
    int layer1Neurons = 27;
	
    float weight = 1./3.8;
	
	network.addNeurons(inputNeurons, decayCurrent, potentialDecay, refractoryPeriod, efficacyDecay, efficacy);
	network.addNeurons(layer1Neurons, decayCurrent, potentialDecay, refractoryPeriod, efficacyDecay, efficacy);
	
	network.allToallConnectivity(&network.getNeuronPopulations()[0], &network.getNeuronPopulations()[1], weight, true, 50);

	// injecting spikes in the input layer
	for (auto idx=0; idx<data[0].size(); idx++)
	{
		network.injectSpike(network.getNeuronPopulations()[0][data[1][idx]].prepareInitialSpike(data[0][idx]));
    }
	
	// injecting the teacher signal for supervised threshold learning
//	network.injectTeacher(&teacher);
	
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
