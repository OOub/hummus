/*
 * test2D.cpp
 * Hummus - spiking neural network simulator
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
	hummus::DataParser dataParser;
	auto trainingData = dataParser.readData("../../data/2Dtest.txt");
	
    //  ----- INITIALISING THE NETWORK -----
	hummus::QtDisplay qtDisplay;
	hummus::Network network(&qtDisplay);
	
	//  ----- CREATING THE NETWORK -----
    network.add2dLayer<hummus::InputNeuron>(0, 2, 8, 8, 2, false, {});
    network.add2dLayer<hummus::LIF>(1, 2, 8, 8, 2, false, {}, true, false, 10, 20, 3, true);
    network.addLayer<hummus::LIF>(1, 1, 1, {});
		
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
