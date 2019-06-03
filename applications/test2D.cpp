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
#include "../source/randomDistributions/normal.hpp"
#include "../source/GUI/qt/qtDisplay.hpp"
#include "../source/neurons/input.hpp"
#include "../source/neurons/LIF.hpp"
#include "../source/synapticKernels/step.hpp"
#include "../source/addons/weightMaps.hpp"
#include "../source/addons/spikeLogger.hpp"
#include "../source/addons/potentialLogger.hpp"
#include "../source/addons/classificationLogger.hpp"

int main(int argc, char** argv) {
	//  ----- READING TRAINING DATA FROM FILE -----
	hummus::DataParser dataParser;
	auto trainingData = dataParser.readData("../../data/2Dtest.txt", true, 50);
	
    //  ----- INITIALISING THE NETWORK -----
    hummus::Network network;
	
    auto& map = network.makeAddon<hummus::WeightMaps>("weightMaps.bin", "../../data/2DtestLabels.txt");
    auto& display = network.makeGUI<hummus::QtDisplay>();
    
	//  ----- CREATING THE NETWORK -----
	auto& step = network.makeSynapticKernel<hummus::Step>();
	
    auto pixel_grid = network.make2dLayer<hummus::Input>(12, 12, 1, {}, nullptr);
    auto convolution = network.makeConvolutionalLayer<hummus::LIF>(network.getLayers()[0], 3, 1, hummus::Normal(), 100, 1, {}, &step, false, 20, 3, true);
    auto pooling = network.makePoolingLayer<hummus::LIF>(network.getLayers()[1], hummus::Normal(), 100, {}, &step, false, 20, 3, true);
    auto output = network.makeLayer<hummus::LIF>(1, {}, &step);
	
    network.allToAll(pooling, output, hummus::Normal());
    
    //  ----- DISPLAY SETTINGS -----
    display.setTimeWindow(100);
    display.trackInputSublayer(0);
    display.trackLayer(1);
    display.trackNeuron(100);
	
    //  ----- RUNNING THE NETWORK -----
    map.activate_for(convolution.neurons);
    network.run(&trainingData, 0.1);

    //  ----- EXITING APPLICATION -----
    return 0;
}
