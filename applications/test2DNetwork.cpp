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
#include "../source/STDP.hpp"

int main(int argc, char** argv)
{
	//  ----- READING TRAINING DATA FROM FILE -----
	adonis_c::DataParser dataParser;
	auto trainingData = dataParser.readData("../../data/2Dtest.txt");
	
    //  ----- INITIALISING THE NETWORK -----
	adonis_c::QtDisplay qtDisplay;
	adonis_c::Network network(&qtDisplay);
	
	//  ----- CREATING THE NETWORK -----
	network.add2dLayer(0, 2, 8, 8, {}, 2, -1, false);
	network.add2dLayer(1, 2, 8, 8, {}, 2, 1, false);
	network.addLayer(2, {}, 1, 1, 1,  false);
		
	network.allToAll(network.getLayers()[0], network.getLayers()[1], 1, 0);
	network.allToAll(network.getLayers()[1], network.getLayers()[2], 1, 0);

	
	//  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(50);
  	qtDisplay.trackInputSublayer(0);
  	qtDisplay.trackLayer(1);
	qtDisplay.trackNeuron(2);
	
    //  ----- RUNNING THE NETWORK -----
    network.run(0.1, &trainingData);

    //  ----- EXITING APPLICATION -----
    return 0;
}
