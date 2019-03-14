/*
 * pokerDVS.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: Spiking neural network classifying the poker-DVS dataset
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/randomDistributions/normal.hpp"
#include "../source/GUI/qtDisplay.hpp"

#include "../source/learningRules/timeInvariantSTDP.hpp"

#include "../source/neurons/LIF.hpp"
#include "../source/neurons/input.hpp"
#include "../source/neurons/decisionMaking.hpp"

#include "../source/addOns/spikeLogger.hpp"
#include "../source/synapticKernels/exponential.hpp"

int main(int argc, char** argv) {
    //  ----- INITIALISING THE NETWORK -----
    hummus::QtDisplay qtDisplay;
	hummus::SpikeLogger spikeLog("pokerSpikeLog.bin");
    hummus::Network network({&spikeLog}, &qtDisplay);

    //  ----- NETWORK PARAMETERS -----
    bool homeostasis = true;
    bool wta = true;
    
    //  ----- CREATING THE NETWORK -----
    auto ti_stdp = network.makeLearningRule<hummus::TimeInvariantSTDP>();
	
    auto kernel = network.makeSynapticKernel<hummus::Step>(5, 1);
	
    network.add2dLayer<hummus::Input>(34, 34, 1, {}, nullptr);
	
    network.addConvolutionalLayer<hummus::LIF>(network.getLayers()[0], 5, 1, hummus::Normal(0.05, 0.01), 100, 1, {}, &kernel, homeostasis, 20, 3, wta);
    network.addPoolingLayer<hummus::LIF>(network.getLayers()[1], hummus::Normal(1, 0), 100, {}, &kernel, homeostasis, 20, 3, wta);
    
    network.addConvolutionalLayer<hummus::LIF>(network.getLayers()[0], 5, 1, hummus::Normal(0.0005, 0.0001), 100, 1, {&ti_stdp}, &kernel, homeostasis, 60, 3, wta);
    network.addPoolingLayer<hummus::LIF>(network.getLayers()[1], hummus::Normal(0.0005, 0.0001), 100, {}, &kernel, homeostasis, 20, 3, wta);
	
    network.addDecisionMakingLayer<hummus::DecisionMaking>("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtestLabel.txt", &kernel, false, {}, 2000, homeostasis, 100, 20, 0, 10, 1, -50);
    
    //  ----- CONNECTING THE NETWORK -----
    network.allToAll(network.getLayers()[2], network.getLayers()[3], hummus::Normal());
    
	//  ----- READING DATA FROM FILE -----
    hummus::DataParser dataParser;
    auto trainingData = dataParser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtest.txt");
    auto testData = dataParser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtest.txt");

	//  ----- DISPLAY SETTINGS -----
    qtDisplay.useHardwareAcceleration(true);
    qtDisplay.setTimeWindow(5000);
    qtDisplay.trackLayer(1);
    qtDisplay.trackNeuron(network.getNeurons().back()->getNeuronID());
    
    network.setVerbose(2);
    
    //  ----- RUNNING THE NETWORK -----
    network.run(&trainingData, 10, &testData);

    //  ----- EXITING APPLICATION -----
    return 0;
}
