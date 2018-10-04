/*
 * receptiveFieldsTest.cpp
 * Adonis_c - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/05/2018
 *
 * Information: Example of a spiking neural network using receptive fields for the pip card task.
 * layer 1 minimum weight -> 19e-10 / 16
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
	auto trainingData = dataParser.readTrainingData("../../data/poker_card_task/2_classes/t10_1pip_2types_200reps.txt");
	
	//  ----- INITIALISING THE NETWORK -----
	adonis_c::QtDisplay qtDisplay;
	adonis_c::SpikeLogger spikeLogger("rfSpikeLog.bin");
	adonis_c::LearningLogger learningLogger("rfLearningLog.bin");
	
	adonis_c::Network network({&spikeLogger, &learningLogger}, &qtDisplay);
	
	//  ----- NETWORK PARAMETERS -----
	float runtime = trainingData.back().timestamp+1;
	float timestep = 0.1;
	int imageSize = 24;
	int rfSize = 4;
	int layer1Neurons = 1;
	int layer1Volume = 5;
	float layer1Weight = 1./5;
	
	float refractoryPeriod = 40;
	float decayCurrent = 10;
	float potentialDecay = 20;
	
	float alpha = 1;
	float lambda = 1;
	
	float eligibilityDecay = 40; // layer 1 temporal window

	bool burstingActivity = false;
	//  ----- INITIALISING THE LEARNING RULE -----
	adonis_c::MyelinPlasticity myelinPlasticity(alpha, lambda);
	
	//  ----- CREATING THE NETWORK -----
	// input layer with 36 receptive fields (2D neurons)
    network.addReceptiveFields(rfSize, imageSize, imageSize, 0, nullptr, -1, decayCurrent, potentialDecay, refractoryPeriod, burstingActivity, eligibilityDecay);
	
	// layer 1 with 36 receptive fields and a layer depth equal to 5 (1D neurons)
	for (auto i=0; i<layer1Volume; i++)
	{
    	network.addReceptiveFields(rfSize, imageSize, imageSize, 1, &myelinPlasticity, layer1Neurons,decayCurrent, potentialDecay, refractoryPeriod, burstingActivity, eligibilityDecay);
    }

    //  ----- CONNECTING THE NETWORK -----
	for (auto& receptiveFieldI: network.getNeuronPopulations())
	{
	    // connecting input layer to layer 1
	    if (receptiveFieldI.layerID == 0)
	    {
			for (auto& receptiveFieldO: network.getNeuronPopulations())
			{
				if (receptiveFieldO.rfID == receptiveFieldI.rfID && receptiveFieldO.layerID != 0)
				{
					network.allToAllConnectivity(&receptiveFieldI.rfNeurons, &receptiveFieldO.rfNeurons, false, layer1Weight, true, 40);
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
	qtDisplay.setTimeWindow(5000);
	qtDisplay.trackLayer(1);
	qtDisplay.trackNeuron(670);

    //  ----- RUNNING THE NETWORK -----
    network.run(runtime, timestep);

    //  ----- EXITING APPLICATION -----
    return 0;
}
