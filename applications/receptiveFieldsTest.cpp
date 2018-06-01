/*
 * ATISNetwork.cpp
 * Baal - clock-driven spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/05/2018
 *
 * Information: Example of a spiking neural network using receptive fields for the pip card task.
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
	auto data = dataParser.readData("../data/pip/1rec_1pip/1pip_1type_200reps.txt");
	
	std::cout << data[0].x << "," << data[0].y << std::endl;
	
	//  ----- INITIALISING THE NETWORK -----
	std::string filename = "rfTest.bin";
	baal::Logger logger(filename);
	baal::Display network({&logger});
	
	//  ----- NETWORK PARAMETERS -----
	float runtime = data.back().timestamp+1;
	float timestep = 1;
	int imageSize = 24;
	int inputlayerRF = 36;
	int layer1RF = 4;
	int layer1Neurons = 10;
	int layer2Neurons = 10;
	int weight = 1;
	
	//  ----- CREATING THE NETWORK -----
	// input layer with 36 receptive fields (2D neurons)
    network.addReceptiveFields(imageSize, inputlayerRF, 0, baal::learningMode::noLearning);
	
	// layer 1 with 4 receptive fields (1D neurons)
    network.addReceptiveFields(imageSize, layer1RF, 1, baal::learningMode::noLearning, layer1Neurons);
	
	// layer 2 with 1 receptive field (1D neurons)
	network.addNeurons(2, baal::learningMode::noLearning, layer2Neurons);
	
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
                    network.allToallConnectivity(&receptiveField.rfNeurons, &network.getNeuronPopulations()[36].rfNeurons, false, weight, false, 0);
                }
                else if (receptiveField.rfNeurons[0].getX() < 12 && receptiveField.rfNeurons[0].getY() >= 12)
                {
                    network.allToallConnectivity(&receptiveField.rfNeurons, &network.getNeuronPopulations()[37].rfNeurons, false, weight, false, 0);
                }
                else if (receptiveField.rfNeurons[0].getX() >= 12 && receptiveField.rfNeurons[0].getY() < 12)
                {
                    network.allToallConnectivity(&receptiveField.rfNeurons, &network.getNeuronPopulations()[38].rfNeurons, false, weight, false, 0);
                }
                else if (receptiveField.rfNeurons[0].getX() >= 12 && receptiveField.rfNeurons[0].getY() >= 12)
                {
                    network.allToallConnectivity(&receptiveField.rfNeurons, &network.getNeuronPopulations()[39].rfNeurons, false, weight, false, 0);
                }
            }
	    }
	    // connecting layer 1 to the output layer
	    else if (receptiveField.layerID == 1)
	    {
	        network.allToallConnectivity(&receptiveField.rfNeurons, &network.getNeuronPopulations().back().rfNeurons, false, weight, false, 0);
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
	network.setTimeWindow(10000);
	network.trackNeuron(-1);
	network.trackLayer(1);
	
    //  ----- RUNNING THE NETWORK -----
    int errorCode = network.run(runtime, timestep);

    //  ----- EXITING APPLICATION -----
    return errorCode;
}
