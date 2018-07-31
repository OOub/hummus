/*
 * unsupervisedNetwork.cpp
 * Adonis_t - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 6/12/2017
 *
 * Information: Example of a basic spiking neural network.
 */

#include <iostream>

#include "../source/neuron.hpp"
#include "../source/dataParser.hpp"
#include "../source/network.hpp"
#include "../source/qtDisplay.hpp"
#include "../source/spikeLogger.hpp"

int main(int argc, char** argv)
{
//  ----- READING DATA FROM FILE -----
	adonis_t::DataParser dataParser;
	
	auto data = dataParser.readData("../../data/generatedPatterns/cleanSignal/0bn0nn4fakePatterns_snnTest_2000reps_10msInterval.txt");
	
//  ----- NETWORK PARAMETERS -----
	adonis_t::QtDisplay qtDisplay;
	adonis_t::SpikeLogger spikeLogger(std::string("loggerTest.bin"));
	adonis_t::Network network({&spikeLogger}, &qtDisplay);

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
	
	float alpha = 1;
	float lambda = 5;
	
	float eligibilityDecay = 20;
	float eligibilityDecay2 = 40;

    float weight = 19e-10/4; //weight dependent on feature size

	network.addNeurons(0, adonis_t::learningMode::noLearning, inputNeurons,decayCurrent, potentialDecay, refractoryPeriod, eligibilityDecay, alpha, lambda);
	network.addNeurons(1, adonis_t::learningMode::delayPlasticityReinforcement, layer1Neurons, decayCurrent, potentialDecay, refractoryPeriod, eligibilityDecay, alpha, lambda);
	network.addNeurons(2, adonis_t::learningMode::delayPlasticityReinforcement, layer2Neurons, decayCurrent2, potentialDecay2, refractoryPeriod, eligibilityDecay2, alpha, lambda);
	
	network.allToallConnectivity(&network.getNeuronPopulations()[0].rfNeurons, &network.getNeuronPopulations()[1].rfNeurons, false, weight, true, 20);
	network.allToallConnectivity(&network.getNeuronPopulations()[1].rfNeurons, &network.getNeuronPopulations()[2].rfNeurons, false, weight, true, 30);
	
	// injecting spikes in the input layer
	for (auto idx=0; idx<data.size(); idx++)
	{
		network.injectSpike(network.getNeuronPopulations()[0].rfNeurons[data[idx].neuronID].prepareInitialSpike(data[idx].timestamp));
    }

//  ----- DISPLAY SETTINGS -----
	qtDisplay.useHardwareAcceleration(true);
	qtDisplay.setTimeWindow(1000);
	qtDisplay.trackNeuron(28);

//  ----- RUNNING THE NETWORK -----
    network.run(runtime, timestep);
	
//  ----- EXITING APPLICATION -----
    return 0;
}
