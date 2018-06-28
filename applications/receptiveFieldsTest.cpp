/*
 * recpetiveFieldsTest.cpp
 * Baal - clock-driven spiking neural network simulator
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
#include "../source/display.hpp"
#include "../source/logger.hpp"
#include "../source/learningLogger.hpp"

int main(int argc, char** argv)
{
	//  ----- READING DATA FROM FILE -----
	baal::DataParser dataParser;
	auto data = dataParser.readData("../../data/pip/2recs_1pip/sst101pip_2types_200reps.txt");
	
	//  ----- INITIALISING THE NETWORK -----
	std::string filename1 = "rfSpikeLog.bin";
	std::string filename2 = "rfLearningLog.bin";
	baal::Logger logger(filename1);
	baal::LearningLogger learningLogger(filename2);
	baal::Display network({&logger, &learningLogger});
	
	//  ----- NETWORK PARAMETERS -----
	float runtime = data.back().timestamp+1;
	float timestep = 0.1;
	int imageSize = 24;
	int inputlayerRF = 36;
	int layer1RF = 36;
	int layer1Neurons = 1;
	float layer1Weight = 19e-10/5;
	
	float refractoryPeriod = 3;
	float decayCurrent = 10;
	float potentialDecay = 20;
	
	float alpha = 1;
	float lambda = 1;
	
	float eligibilityDecay = 40; // layer 1 temporal window
	
	//  ----- CREATING THE NETWORK -----
	// input layer with 36 receptive fields (2D neurons)
    network.addReceptiveFields(inputlayerRF, 0, baal::learningMode::noLearning, imageSize, -1, decayCurrent, potentialDecay, refractoryPeriod, eligibilityDecay, alpha, lambda);
	
	// layer 1 with 36 receptive fields and a layer depth equal to 5 (1D neurons)
	for (auto i=0; i<5; i++)
	{
    	network.addReceptiveFields(layer1RF, 1, baal::learningMode::delayPlasticityNoReinforcement, imageSize, layer1Neurons,decayCurrent, potentialDecay, refractoryPeriod, eligibilityDecay, alpha, lambda);
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
					network.allToallConnectivity(&receptiveFieldI.rfNeurons, &receptiveFieldO.rfNeurons, false, layer1Weight, true, 40);
				}
			}
	    }
	}

	//  ----- INJECTING SPIKES -----
	for (auto& event: data)
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
	network.useHardwareAcceleration(true);
	network.setTimeWindow(5000);
	network.trackLayer(1);
	network.trackNeuron(670);

    //  ----- RUNNING THE NETWORK -----
    int errorCode = network.run(runtime, timestep);

    //  ----- EXITING APPLICATION -----
    return errorCode;
}
