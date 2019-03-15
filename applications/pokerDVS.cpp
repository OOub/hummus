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

#include "../source/learningRules/stdp.hpp"
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
    bool conv_wta = false;
    bool pool_wta = true;
    
    //  ----- CREATING THE NETWORK -----
    auto ti_stdp = network.makeLearningRule<hummus::TimeInvariantSTDP>(1, -1, 1, -1);
    auto stdp = network.makeLearningRule<hummus::STDP>();
    
    auto kernel = network.makeSynapticKernel<hummus::Step>(5);
	
    network.add2dLayer<hummus::Input>(34, 34, 1, {}, nullptr);
	
    network.addConvolutionalLayer<hummus::LIF>(network.getLayers()[0], 5, 1, hummus::Normal(0.8, 0.1, 5, 3), 80, 1, {&stdp}, &kernel, homeostasis, 20, 3, conv_wta);
    network.addPoolingLayer<hummus::LIF>(network.getLayers()[1], hummus::Normal(1, 0), 100, {}, &kernel, homeostasis, 20, 3, pool_wta);
    
    network.addConvolutionalLayer<hummus::LIF>(network.getLayers()[0], 5, 1, hummus::Normal(0.8, 0.1, 5, 3), 80, 1, {&stdp}, &kernel, homeostasis, 60, 3, conv_wta);
    network.addPoolingLayer<hummus::LIF>(network.getLayers()[1], hummus::Normal(1, 0), 100, {}, &kernel, homeostasis, 60, 3, pool_wta);
	
    network.addDecisionMakingLayer<hummus::DecisionMaking>("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtrainingLabel.txt", &kernel, false, {}, 2000, homeostasis, 80);
    
    //  ----- CONNECTING THE NETWORK -----
    network.lateralInhibition(network.getLayers()[1], -0.6);
    network.lateralInhibition(network.getLayers()[3], -0.6);
    network.allToAll(network.getLayers()[4], network.getLayers()[5], hummus::Normal(0.8, 0.1));
    
	//  ----- READING DATA FROM FILE -----
    hummus::DataParser dataParser;
    auto trainingData = dataParser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtraining.txt");
    auto testData = dataParser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtest.txt");

	//  ----- DISPLAY SETTINGS -----
    qtDisplay.useHardwareAcceleration(true);
    qtDisplay.setTimeWindow(5000);
    qtDisplay.trackLayer(5);
    qtDisplay.trackNeuron(network.getNeurons().back()->getNeuronID());
    
    //  ----- RUNNING THE NETWORK -----
    network.run(&trainingData, 0, &testData);

    //  ----- EXITING APPLICATION -----
    return 0;
}
