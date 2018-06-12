/*
 * ATISNetwork.cpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/05/2018
 *
 * Information: Example of a spiking neural network using receptive fields for the pip card task.
 * layer 1 minimum weight -> 19e-10 / 16
 * layer 2 minimum weight -> 19e-10 / 64
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
	auto data = dataParser.readData("../../data/pip/2recs_1pip/sst101pip_2types_200reps.txt");
	
	//  ----- INITIALISING THE NETWORK -----
	std::string filename = "rfTest.bin";
	baal::Logger logger(filename);
	baal::Display network({&logger});
	
	//  ----- NETWORK PARAMETERS -----
	float runtime = data.back().timestamp+1;
	float timestep = 0.1;
	int imageSize = 24;
	int inputlayerRF = 36;
	int layer1RF = 4;
	int layer1Neurons = 20;
	int layer2Neurons = 20;
	
	float layer1Weight = 19e-10/8; // can be maximum 16
	float layer2Weight = 19e-10/32; // can be maximum 64
	
	float refractoryPeriod = 3;
	float decayCurrent = 10;
	float potentialDecay = 20;
	
	float decayCurrent2 = 40;
	float potentialDecay2 = 50;
	
	float alpha = 1; // check by how much it's changing
	float lambda = 1;
	
	float eligibilityDecay = 50; // layer 1 temporal window
	float eligibilityDecay2 = 100; // layer 2 temporal window
	
	//  ----- CREATING THE NETWORK -----
	// input layer with 36 receptive fields (2D neurons)
    network.addReceptiveFields(inputlayerRF, 0, baal::learningMode::noLearning, imageSize, -1, decayCurrent, potentialDecay, refractoryPeriod, eligibilityDecay, alpha, lambda);
	
	// layer 1 with 4 receptive fields (1D neurons)
    network.addReceptiveFields(layer1RF, 1, baal::learningMode::delayPlasticityNoReinforcement, imageSize, layer1Neurons,decayCurrent, potentialDecay, refractoryPeriod, eligibilityDecay, alpha, lambda);
	
	// layer 2 with 1 receptive field (1D neurons)
	network.addNeurons(2, baal::learningMode::delayPlasticityNoReinforcement, layer2Neurons,decayCurrent2, potentialDecay2, refractoryPeriod, eligibilityDecay2, alpha, lambda);
	
    //  ----- CONNECTING THE NETWORK -----
	for (auto& receptiveField: network.getNeuronPopulations())
	{
	    // connecting input layer to layer 1
	    if (receptiveField.layerID == 0)
	    {
	        if (receptiveField.rfNeurons[0].getX() >= 0 && receptiveField.rfNeurons[0].getY() >= 0)
	        {
                if (receptiveField.rfNeurons[0].getX() < 12 && receptiveField.rfNeurons[0].getY() < 12)
                {
                    network.allToallConnectivity(&receptiveField.rfNeurons, &network.getNeuronPopulations()[36].rfNeurons, false, layer1Weight, true, 50);
                }
                else if (receptiveField.rfNeurons[0].getX() < 12 && receptiveField.rfNeurons[0].getY() >= 12)
                {
                    network.allToallConnectivity(&receptiveField.rfNeurons, &network.getNeuronPopulations()[37].rfNeurons, false, layer1Weight, true, 50);
                }
                else if (receptiveField.rfNeurons[0].getX() >= 12 && receptiveField.rfNeurons[0].getY() < 12)
                {
                    network.allToallConnectivity(&receptiveField.rfNeurons, &network.getNeuronPopulations()[38].rfNeurons, false, layer1Weight, true, 50);
                }
                else if (receptiveField.rfNeurons[0].getX() >= 12 && receptiveField.rfNeurons[0].getY() >= 12)
                {
                    network.allToallConnectivity(&receptiveField.rfNeurons, &network.getNeuronPopulations()[39].rfNeurons, false, layer1Weight, true, 50);
                }
            }
	    }
	    // connecting layer 1 to the output layer
	    else if (receptiveField.layerID == 1)
	    {
	        network.allToallConnectivity(&receptiveField.rfNeurons, &network.getNeuronPopulations().back().rfNeurons, false, layer2Weight, true, 100);
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
