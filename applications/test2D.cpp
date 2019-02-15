/*
 * test2D.cpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: Example of a basic spiking neural network.
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/GUI/qtDisplay.hpp"
#include "../source/neurons/inputNeuron.hpp"
#include "../source/neurons/LIF.hpp"

int main(int argc, char** argv) {
	//  ----- READING TRAINING DATA FROM FILE -----
	adonis::DataParser dataParser;
	auto trainingData = dataParser.readData("../../data/2Dtest.txt");
	
    //  ----- INITIALISING THE NETWORK -----
	adonis::QtDisplay qtDisplay;
	adonis::Network network(&qtDisplay);
	
	//  ----- CREATING THE NETWORK -----
    network.add2dLayer<adonis::InputNeuron>(0, 2, 8, 8, 2, false, {});
    network.add2dLayer<adonis::LIF>(1, 2, 8, 8, 2, false, {}, false, 100, 10, 20, 3, true);
    network.addLayer<adonis::LIF>(1, 1, 1, {});
		
	network.convolution(network.getLayers()[0], network.getLayers()[1], 1./2, 0);
	network.allToAll(network.getLayers()[1], network.getLayers()[2], 1., 0);
	
	//  ----- DISPLAY SETTINGS -----
  	qtDisplay.useHardwareAcceleration(true);
  	qtDisplay.setTimeWindow(100);
  	qtDisplay.trackInputSublayer(0);
  	qtDisplay.trackLayer(1);
	qtDisplay.trackNeuron(128);
	
    //  ----- RUNNING THE NETWORK -----
    network.run(&trainingData, 0.1);

    //  ----- EXITING APPLICATION -----
    return 0;
}
