/*
 * supervisedNetwork.cpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 09/01/2018
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
	int repeatsInTeacher = 300;
	baal::DataParser dataParser;
	
	// time jitter test
	auto data = dataParser.readData("../data/generatedPatterns/timeJitter/3timeJitter0bn0nn4fakePatterns_snnTest_400reps_10msInterval.txt");
	
	// supervised learning
	auto teacher = dataParser.readData("../data/generatedPatterns/timeJitter/3teacherSignal.txt");
	
	for (auto idx=0; idx<teacher.size(); idx++)
	{
		teacher.resize(repeatsInTeacher);
	}

//  ----- NETWORK PARAMETERS -----
	std::string filename = "supervisedLearning_3jitter.bin";

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
    int layer1Neurons = 10;
	int layer2Neurons = 4;
	
    float weight = 19e-10/10;
	float alpha = 1;
	float lambda = 1;

	network.addNeurons(0, baal::learningMode::noLearning, inputNeurons, decayCurrent, potentialDecay, refractoryPeriod, alpha, lambda);
	network.addNeurons(1, baal::learningMode::delayPlasticity, layer1Neurons, decayCurrent, potentialDecay, refractoryPeriod, alpha, lambda);
	network.addNeurons(2, baal::learningMode::delayPlasticity, layer2Neurons, decayCurrent2, potentialDecay2, refractoryPeriod, alpha, lambda);
	
	network.allToallConnectivity(&network.getNeuronPopulations()[0].rfNeurons, &network.getNeuronPopulations()[1].rfNeurons, false, weight, true, 20);
	network.allToallConnectivity(&network.getNeuronPopulations()[1].rfNeurons, &network.getNeuronPopulations()[2].rfNeurons, false,  19e-10/5, true, 20);
	
	// injecting spikes in the input layer
	for (auto idx=0; idx<data.size(); idx++)
	{
		network.injectSpike(network.getNeuronPopulations()[0].rfNeurons[data[idx].neuronID].prepareInitialSpike(data[idx].timestamp));
    }

	// injecting the teacher signal for supervised threshold learning
  	network.injectTeacher(&teacher);

//  ----- DISPLAY SETTINGS -----
	network.useHardwareAcceleration(true);
	network.setTimeWindow(1000);
	network.trackNeuron(28);

//  ----- RUNNING THE NETWORK -----
    int errorCode = network.run(runtime, timestep);

//  ----- EXITING APPLICATION -----
    return errorCode;
}
