/*
 * smallNetwork.cpp
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
	int repeatsInTeacher = 50;
	baal::DataParser dataParser;
	
	// clean signal test
//    auto data = dataParser.read1D("../../data/generatedPatterns/cleanSignal/0bn0nn4fakePatterns_snnTest_2000reps_10msInterval.txt");
	
	// time jitter test
	auto data = dataParser.read1D("../../data/generatedPatterns/timeJitter/1.5timeJitter0bn0nn4fakePatterns_snnTest_2000reps_10msInterval.txt");
//	auto data = dataParser.read1D("../../data/generatedPatterns/timeJitter/3.5timeJitter0bn0nn4fakePatterns_snnTest_2000reps_10msInterval.txt");
//	auto data = dataParser.read1D("../../data/generatedPatterns/timeJitter/5.5timeJitter0bn0nn4fakePatterns_snnTest_2000reps_10msInterval.txt");
//	auto data = dataParser.read1D("../../data/generatedPatterns/timeJitter/7.5timeJitter0bn0nn4fakePatterns_snnTest_2000reps_10msInterval.txt");
//	auto data = dataParser.read1D("../../data/generatedPatterns/timeJitter/9.5timeJitter0bn0nn4fakePatterns_snnTest_2000reps_10msInterval.txt");
	
	// additive noise test
//	auto data = dataParser.read1D("../../data/generatedPatterns/additiveNoise/10bn0nn4fakePatterns_snnTest_2000reps_10msInterval.txt");
	
	// noisy signal
//	auto data = dataParser.read1D("../../data/generatedPatterns/noisySignal/1.5timeJitter10bn0nn4fakePatterns_snnTest_2000reps_10msInterval.txt");
	
	// supervised learning
//	auto teacher = dataParser.read1D("../../data/generatedPatterns/cleanSignal/clean_teacherSignal.txt");
	auto teacher = dataParser.read1D("../../data/generatedPatterns/timeJitter/1.5teacherSignal.txt");
//	auto teacher = dataParser.read1D("../../data/generatedPatterns/timeJitter/3.5teacherSignal.txt");
//	auto teacher = dataParser.read1D("../../data/generatedPatterns/timeJitter/5.5teacherSignal.txt");
//	auto teacher = dataParser.read1D("../../data/generatedPatterns/timeJitter/7.5teacherSignal.txt");
//	auto teacher = dataParser.read1D("../../data/generatedPatterns/timeJitter/9.5teacherSignal.txt");

	teacher.resize(repeatsInTeacher);
	
//  ----- NETWORK PARAMETERS -----
	
//	std::string filename = "supervisedLearning_clean.bin";
	std::string filename = "supervisedLearning_jitter.bin";
//	std::string filename = "supervisedLearning_additive.bin";
	baal::Logger logger(filename);
	baal::Display network({&logger});
	
	network.learningLogger("learningLog.txt");
	
//  ----- INITIALISING THE NETWORK -----
	float runtime = data[0].back()+1;
	float timestep = 0.1;
	
	float decayCurrent = 10;
	float potentialDecay = 20;
	float refractoryPeriod = 3;
    float efficacyDecay = 1000;
    float efficacy = 1;
	
    int inputNeurons = 27;
    int layer1Neurons = 27;
	
    float weight = 19e-10/10;
	
	network.addNeurons(inputNeurons, decayCurrent, potentialDecay, refractoryPeriod, efficacyDecay, efficacy);
	network.addNeurons(layer1Neurons, decayCurrent, potentialDecay, refractoryPeriod, efficacyDecay, efficacy);
	
	network.allToallConnectivity(&network.getNeuronPopulations()[0], &network.getNeuronPopulations()[1], false, weight, true, 20);

	network.getNeuronPopulations()[1][data[1][1]].potentialLogger("potentialLog.txt");
	
	// injecting spikes in the input layer
	for (auto idx=0; idx<data[0].size(); idx++)
	{
		network.injectSpike(network.getNeuronPopulations()[0][data[1][idx]].prepareInitialSpike(data[0][idx]));
    }
	
	// injecting the teacher signal for supervised threshold learning
  	network.injectTeacher(&teacher);
	
//  ----- DISPLAY SETTINGS -----
	network.useHardwareAcceleration(true);
	network.setTimeWindow(1000);
	network.setOutputMinY(layer1Neurons);
	network.trackNeuron(30);
	
//  ----- RUNNING THE NETWORK -----
    int errorCode = network.run(runtime, timestep);
	
//  ----- EXITING APPLICATION -----
    return errorCode;
}
