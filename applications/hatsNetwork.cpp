/*
 * hatsNetwork.cpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/05/2018
 *
 * Information: Example of a basic spiking neural network.
 */

#include <iostream>

#include "../source/network.hpp"
#include "../source/qtDisplay.hpp"
#include "../source/spikeLogger.hpp"
#include "../source/stdp.hpp"

int main(int argc, char** argv)
{

    //  ----- READING TRAINING DATA FROM FILE -----
	adonis_c::DataParser dataParser;
	auto trainingData = dataParser.readTrainingData("../../data/hats/poisson/nCars_training_all_1rep_poisson.txt");
	
    //  ----- INITIALISING THE NETWORK -----
	adonis_c::QtDisplay qtDisplay;
	adonis_c::SpikeLogger spikeLogger("hatsPoissonSpikeLog.bin");
	adonis_c::Network network({&spikeLogger}, &qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	float runtime = trainingData.back().timestamp+1;
	float timestep = 0.1;
	
	int gridWidth = 42;
	int gridHeight = 35;
	int rfSize = 1;
	
	float decayCurrent = 10;
	float potentialDecay = 20;
	float refractoryPeriod = 3;
	
    int layer1Neurons = 1;
    float weight = 1./5;

	float eligibilityDecay = 100; // temporal window
	
	//  ----- INITIALISING THE LEARNING RULE -----
	adonis_c::Stdp stdp(1, 1, 20, 20);
	
	//  ----- CREATING THE NETWORK -----
	// Input layer (2D neurons)
	network.addReceptiveFields(rfSize, gridWidth, gridHeight, 0, &stdp, -1, decayCurrent, potentialDecay, refractoryPeriod, false, eligibilityDecay);

	// Hidden layer 1
	network.addReceptiveFields(rfSize, gridWidth, gridHeight, 1, &stdp, layer1Neurons, decayCurrent, potentialDecay, refractoryPeriod, false, eligibilityDecay);

	// Output layer
	network.addNeurons(2);
	
    //  ----- CONNECTING THE NETWORK -----
	// input layer -> hidden layer 1
	for (auto& receptiveFieldI: network.getNeuronPopulations())
	{
	    // connecting input layer to layer 1
	    if (receptiveFieldI.layerID == 0)
	    {
			for (auto& receptiveFieldO: network.getNeuronPopulations())
			{
				if (receptiveFieldO.rfID == receptiveFieldI.rfID && receptiveFieldO.layerID == 1)
				{
					network.allToAllConnectivity(&receptiveFieldI.rfNeurons, &receptiveFieldO.rfNeurons, false, weight, false, 0);
				}
			}
	    }
	}

	// hidden layer 1 -> output layer
	network.allToAllConnectivity(&network.getNeuronPopulations()[1].rfNeurons, &network.getNeuronPopulations()[2].rfNeurons, false, 1, false, 0);
	
    //  ----- INJECTING SPIKES -----
	for (auto& event: trainingData)
	{
	    for (auto& receptiveField: network.getNeuronPopulations())
	    {
	   	    if (receptiveField.layerID == 0)
	        {
	            for (auto& neuron: receptiveField.rfNeurons)
	            {
	                if (neuron.getX() == event.x && neuron.getY() == event.y)
	                {
                        network.injectSpike(neuron.prepareInitialSpike(event.timestamp));
	                    break;
	                }
	            }
	        }
	    }
	}
	
    //  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(1000);
  	qtDisplay.trackLayer(2);
	
    //  ----- RUNNING THE NETWORK -----
    network.run(runtime, timestep);

    //  ----- EXITING APPLICATION -----
    return 0;
}
