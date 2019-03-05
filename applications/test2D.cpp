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
#include "../source/rand.hpp"
#include "../source/GUI/qtDisplay.hpp"
#include "../source/neurons/inputNeuron.hpp"
#include "../source/neurons/LIF.hpp"

int main(int argc, char** argv) {
	//  ----- READING TRAINING DATA FROM FILE -----
	hummus::DataParser dataParser;
	auto trainingData = dataParser.readData("../../data/2Dtest.txt", true, 50);
	
    //  ----- INITIALISING THE NETWORK -----
	hummus::QtDisplay qtDisplay;
	hummus::Network network(&qtDisplay);
	
	//  ----- CREATING THE NETWORK -----
    network.add2dLayer<hummus::InputNeuron>(12, 12, 2, {});
    network.addConvolutionalLayer<hummus::LIF>(network.getLayers()[0], 3, 3, hummus::Rand(), 100, 1, {}, false, false, 10, 20, 3, true);
    network.addPoolingLayer<hummus::LIF>(network.getLayers()[1], hummus::Rand(), 100, {}, false, false, 10, 20, 3, true);
    network.addLayer<hummus::LIF>(1, {});
    
    network.allToAll(network.getLayers()[2], network.getLayers()[3], hummus::Rand());
    
    //  ----- DISPLAY SETTINGS -----
    qtDisplay.useHardwareAcceleration(true);
    qtDisplay.setTimeWindow(100);
    qtDisplay.trackInputSublayer(0);
    qtDisplay.trackLayer(1);
    qtDisplay.trackNeuron(100);

    //  ----- RUNNING THE NETWORK -----
    network.run(&trainingData, 0);

    //  ----- EXITING APPLICATION -----
    return 0;
}
