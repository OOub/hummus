/*
 * hatsNetwork.cpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 09/10/2018
 *
 * Information: Example of a basic spiking neural network.
 
 I think I'm connecting them wrong
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
    // IDs for each layer (the order is very important for the learning rule so to avoid mistakes we create variables for the IDs that will be used wherever required)
    int layer0 = 0;
    int layer1 = 1;
    int layer2 = 2;
	
	float runtime = trainingData.back().timestamp+1;
	float timestep = 0.1;
	
	int gridWidth = 42;
	int gridHeight = 35;
	int rfSize = 7;
	
	float decayCurrent = 10;
	float potentialDecay = 20;
	float refractoryPeriod = 3;

	float eligibilityDecay = 20;
	
	//  ----- INITIALISING THE LEARNING RULE -----
	adonis_c::Stdp stdp(layer1, layer2);
	
	//  ----- CREATING THE NETWORK -----
	// Input layer (2D neurons)
	network.addReceptiveFields(rfSize, gridWidth, gridHeight, layer0, nullptr, -1, decayCurrent, potentialDecay, refractoryPeriod, false, eligibilityDecay);
	
	// Hidden layer 1
	network.addReceptiveFields(rfSize, gridWidth, gridHeight, layer1, &stdp, 1, decayCurrent, potentialDecay, refractoryPeriod, false, eligibilityDecay);

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
				if (receptiveFieldO.rfID == receptiveFieldI.rfID && receptiveFieldO.layerID == 1)
				{
					network.allToAllConnectivity(&receptiveFieldI.rfNeurons, &receptiveFieldO.rfNeurons, false, 1, false, 0);
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
					network.allToAllConnectivity(&receptiveFieldI.rfNeurons, &receptiveFieldO.rfNeurons, false, 1, false, 0);
				}
			}
	    }
	}
	
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
  	qtDisplay.trackLayer(1);
  	qtDisplay.trackNeuron(1500);
	
    //  ----- RUNNING THE NETWORK -----
    network.run(runtime, timestep);

    //  ----- EXITING APPLICATION -----
    return 0;
}
