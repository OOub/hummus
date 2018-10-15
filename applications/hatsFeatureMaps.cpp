/*
 * hatsNetwork.cpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 09/10/2018
 *
 * Information: Spiking neural network running with histograms of averaged time surfaces converted into spikes.
 */

#include <iostream> 

#include "../source/network.hpp"
#include "../source/qtDisplay.hpp"
#include "../source/spikeLogger.hpp"
#include "../source/stdp.hpp"

int main(int argc, char** argv)
{
    //  ----- INITIALISING THE NETWORK -----
	adonis_c::QtDisplay qtDisplay;
	adonis_c::Network network(&qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	
    // IDs for each layer (order is important)
    int layer0 = 0;
    int layer1 = 1;
    int layer2 = 2;
	
	int gridWidth = 42;
	int gridHeight = 35;
	int rfSize = 7;
	
	float decayCurrent = 5;
	float potentialDecay = 10;
	float refractoryPeriod = 3;

	float eligibilityDecay = 10;
	
	//  ----- INITIALISING THE LEARNING RULE -----
	adonis_c::Stdp stdp(layer1, layer2);
	
	//  ----- CREATING THE NETWORK -----
	// Input layer (2D neurons)
	network.addReceptiveFields(rfSize, gridWidth, gridHeight, layer0, nullptr, -1, decayCurrent, potentialDecay, refractoryPeriod, false, eligibilityDecay);
	
	// Hidden layer 1
	network.addNeurons(layer1, &stdp, 10, decayCurrent, potentialDecay, refractoryPeriod, false, eligibilityDecay);

	// Output layer
	network.addNeurons(layer2, &stdp, 1, decayCurrent, potentialDecay, refractoryPeriod, false, eligibilityDecay);
	
    //  ----- CONNECTING THE NETWORK -----
	// input layer -> hidden layer 1
	for (auto& receptiveFieldI: network.getNeuronPopulations())
	{
	    // connecting input layer to layer 1
	    if (receptiveFieldI.layerID == 0)
	    {
			for (auto& receptiveFieldO: network.getNeuronPopulations())
			{
				if (receptiveFieldO.layerID == 1)
				{
					network.allToAllConnectivity(&receptiveFieldI.rfNeurons, &receptiveFieldO.rfNeurons, true, 1./30, false, 0);
				}
			}
	    }
	}

	// hidden layer 1 -> output layer
	for (auto& receptiveFieldI: network.getNeuronPopulations())
	{
	    // connecting input layer to layer 1
	    if (receptiveFieldI.layerID == 1)
	    {
			for (auto& receptiveFieldO: network.getNeuronPopulations())
			{
				if (receptiveFieldO.layerID == 2)
				{
					network.allToAllConnectivity(&receptiveFieldI.rfNeurons, &receptiveFieldO.rfNeurons, true, 1./5, false, 0);
				}
			}
	    }
	}
	
	//  ----- READING TRAINING DATA FROM FILE -----
	adonis_c::DataParser dataParser;
    auto trainingData = dataParser.readTrainingData("../../data/hats/poisson/nCars_train_10samplePerc_1rep.txt");
	
    //  ----- INJECTING TRAINING SPIKES -----
	network.injectSpikeFromData(&trainingData);
	
	//  ----- READING TEST DATA FROM FILE -----
	auto testingData = dataParser.readTestData(&network, "../../data/hats/poisson/nCars_test_10samplePerc_1rep.txt");
	
	//  ----- INJECTING TEST SPIKES -----
	network.injectSpikeFromData(&testingData);
	
	//  ----- ADDING THE LABELS -----
	auto labels = dataParser.readLabels("../../data/hats/poisson/nCars_train_10samplePerc_1repLabel.txt", "../../data/hats/poisson/nCars_test_10samplePerc_1repLabel.txt");
	network.addLabels(&labels);
	
    //  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(1000);
  	qtDisplay.trackLayer(2);
  	qtDisplay.trackNeuron(1500);
	
    //  ----- RUNNING THE NETWORK -----
    float runtime = trainingData.back().timestamp+testingData.back().timestamp+1;
	float timestep = 0.1;
	
    network.run(runtime, timestep);

    //  ----- EXITING APPLICATION -----
    return 0;
}
