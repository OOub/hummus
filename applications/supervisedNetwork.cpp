/*
 * supervisedNetwork.cpp
 * Adonis_c - clock-driven spiking neural network simulator
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
#include "../source/qtDisplay.hpp"
#include "../source/spikeLogger.hpp"
#include "../source/learningLogger.hpp"
#include "../source/myelinPlasticity.hpp"

int main(int argc, char** argv)
{
    //  ----- READING TRAINING DATA FROM FILE -----
	adonis_c::DataParser dataParser;
	
	auto trainingData = dataParser.readTrainingData("../../data/1D_patterns/control/oneD_10neurons_4patterns.txt");
	auto teacher = dataParser.readTeacherSignal("../../data/1D_patterns/control/oneD_10neurons_4patterns_teacherSignal.txt");
	
    //  ----- INITIALISING THE NETWORK -----
	adonis_c::QtDisplay qtDisplay;
	adonis_c::SpikeLogger spikeLogger("10neurons_4patterns_supervised_spikeLog.bin");
	adonis_c::LearningLogger learningLogger("10neurons_4patterns_supervised_learningLog.bin");
	adonis_c::Network network({&spikeLogger, &learningLogger}, &qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	float runtime = trainingData.back().timestamp+1;
	float timestep = 0.1;

	float decayCurrent = 10;
	float potentialDecay = 20;
	
	float refractoryPeriod = 3;

    int inputNeurons = 10;
    int layer1Neurons = 10;
	
	float alpha = 1;
	float lambda = 0.1;
	float eligibilityDecay = 20;
    float weight = 19e-10/10;

	//  ----- INITIALISING THE LEARNING RULE -----
	adonis_c::MyelinPlasticity myelinPlasticity(alpha, lambda);
	
    //  ----- CREATING THE NETWORK -----
	network.addNeurons(0, nullptr, inputNeurons, decayCurrent, potentialDecay, refractoryPeriod, eligibilityDecay);
	network.addNeurons(1, &myelinPlasticity, layer1Neurons, decayCurrent, potentialDecay, refractoryPeriod, eligibilityDecay);
	
	//  ----- CONNECTING THE NETWORK -----
	network.allToAllConnectivity(&network.getNeuronPopulations()[0].rfNeurons, &network.getNeuronPopulations()[1].rfNeurons, false, weight, true, 10);
	
	//  ----- INJECTING SPIKES -----
	for (auto idx=0; idx<trainingData.size(); idx++)
	{
		network.injectSpike(network.getNeuronPopulations()[0].rfNeurons[trainingData[idx].neuronID].prepareInitialSpike(trainingData[idx].timestamp));
    }

	// injecting the teacher signal for supervised threshold learning
  	network.injectTeacher(&teacher);

    //  ----- DISPLAY SETTINGS -----
	qtDisplay.useHardwareAcceleration(true);
	qtDisplay.setTimeWindow(1000);
	qtDisplay.trackNeuron(10);
	
	// to turn off learning and start testing
	network.turnOffLearning(10000);
	
    //  ----- RUNNING THE NETWORK -----
    network.run(runtime, timestep);

    //  ----- EXITING APPLICATION -----
    return 0;
}
