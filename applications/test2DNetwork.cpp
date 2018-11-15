/*
 * testNetwork.cpp
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
	auto trainingData = dataParser.readTrainingData("../../data/2Dtest.txt");
	
    //  ----- INITIALISING THE NETWORK -----
	adonis_c::QtDisplay qtDisplay;
	adonis_c::Network network(&qtDisplay);
	
    //  ----- NETWORK PARAMETERS -----
	float runtime = 40;
	float timestep = 0.1;

	//  ----- INITIALISING THE LEARNING RULE -----
	adonis_c::Stdp stdp(0, 1);
	
	//  ----- CREATING THE NETWORK -----
	network.add2dLayer(0, 2, 8, 8, nullptr, 2);
	network.add2dLayer(1, 2, 8, 8, nullptr, 2, 1);
	network.addLayer(2, nullptr, 1, 1, 1);
		
	network.convolution(network.getLayers()[0], network.getLayers()[1], false, 1., false, 0);
	network.allToAll(network.getLayers()[1], network.getLayers()[2], false, 1., false, 0);
	
    //  ----- INJECTING SPIKES -----	
	network.injectSpikeFromData(&trainingData);

	//  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(false);
  	qtDisplay.setTimeWindow(runtime);
  	qtDisplay.trackLayer(1);
	qtDisplay.trackNeuron(network.getNeurons().back().getNeuronID());
	
    //  ----- RUNNING THE NETWORK -----
    network.run(runtime, timestep);

    //  ----- EXITING APPLICATION -----
    return 0;
}
