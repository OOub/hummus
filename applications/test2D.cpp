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
    
	//  ----- CREATING THE NEURONS -----
    auto pixel_grid = network.makeGrid<hummus::Input>(12, 12, 1, {});
    auto convolution = network.makeGrid<hummus::LIF>(pixel_grid, 1, 3, 1, {}, false, 20, 10, 3, true);
    auto pooling = network.makeSubsampledGrid<hummus::LIF>(convolution, {}, false, 20, 10, 3, true);
    auto output = network.makeLayer<hummus::LIF>(1, {});
	
    //  ----- CONNECTING THE NEURONS -----
    network.convolution<hummus::Exponential>(pixel_grid, convolution, 1, hummus::Normal(), 100);
    network.pooling<hummus::Exponential>(convolution, pooling, 1, hummus::Normal(), 100);
    network.allToAll<hummus::Exponential>(pooling, output, 1, hummus::Normal(), 100);
    
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
