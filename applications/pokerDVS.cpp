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
#include "../source/addOns/classificationLogger.hpp"
#include "../source/synapticKernels/exponential.hpp"

int main(int argc, char** argv) {
    
    bool deepSpiking = false; // choose between feedforward or deep spiking neural network
    
    //  ----- INITIALISING THE NETWORK -----
    hummus::QtDisplay qtDisplay;
    hummus::ClassificationLogger classifLog("pokerClassifLog.bin");
	hummus::SpikeLogger spikeLog("pokerSpikeLog.bin");
    hummus::Network network({&spikeLog, &classifLog}, &qtDisplay);
    
    auto ti_stdp = network.makeLearningRule<hummus::TimeInvariantSTDP>(); // time-invariant STDP learning rule
    auto kernel = network.makeSynapticKernel<hummus::Exponential>(5); // exponential synaptic kernel
    
    if (deepSpiking) {
        //  ----- DEEP SPIKING NEURAL NETWORK -----
        
        /// parameters
        bool homeostasis = true;
        bool conv_wta = false;
        bool pool_wta = true;

        /// creating the layers
        network.add2dLayer<hummus::Input>(34, 34, 1, {}, nullptr); // input layer
        network.addConvolutionalLayer<hummus::LIF>(network.getLayers()[0], 5, 1, hummus::Normal(0.8, 0.1, 5, 3), 80, 4, {&ti_stdp}, &kernel, homeostasis, 20, 3, conv_wta); // first convolution
        network.addPoolingLayer<hummus::LIF>(network.getLayers()[1], hummus::Normal(1, 0), 100, {}, &kernel, homeostasis, 20, 3, pool_wta); // first pooling
        network.addConvolutionalLayer<hummus::LIF>(network.getLayers()[0], 5, 1, hummus::Normal(0.8, 0.1, 5, 3), 80, 8, {&ti_stdp}, &kernel, homeostasis, 60, 3, conv_wta); // second convolution
        network.addPoolingLayer<hummus::LIF>(network.getLayers()[1], hummus::Normal(1, 0), 100, {}, &kernel, homeostasis, 60, 3, pool_wta); // second pooling
        network.addDecisionMakingLayer<hummus::DecisionMaking>("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtrainingLabel.txt", &kernel, false, {}, 2500, homeostasis, 80); // decision-making
        
        /// connecting the layers
        network.lateralInhibition(network.getLayers()[1], -0.6);
        network.lateralInhibition(network.getLayers()[3], -0.6);
        network.allToAll(network.getLayers()[4], network.getLayers()[5], hummus::Normal(0.8, 0.1));
    } else {
        // ----- SIMPLE FEEDFORWARD -----
        
        /// parameters
        bool homeostasis = true;
        bool wta = true;
        bool burst = true;
        
        /// creating the layers
        network.add2dLayer<hummus::Input>(34, 34, 1, {}, nullptr); // input layer
        network.addLayer<hummus::LIF>(100, {}, &kernel, homeostasis, 20, 3, wta, burst); // hidden layer with STDP
        network.addLayer<hummus::LIF>(2, {}, &kernel, homeostasis, 40, 2500, true, burst); // output layer with 2 neurons
        
        /// connecting the layers
        network.allToAll(network.getLayers()[0], network.getLayers()[1], hummus::Normal(0.05, 0.1, 5, 3), 80);
        network.allToAll(network.getLayers()[1], network.getLayers()[2], hummus::Normal(0.8, 0.1));
    }
    
	//  ----- READING DATA FROM FILE -----
    hummus::DataParser dataParser;
    auto trainingData = dataParser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtraining.txt");
    auto testData = dataParser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtest.txt");

	//  ----- DISPLAY SETTINGS -----
    qtDisplay.useHardwareAcceleration(true);
    qtDisplay.setTimeWindow(20000);
    qtDisplay.trackLayer(1);
    qtDisplay.trackNeuron(network.getNeurons().back()->getNeuronID());
    
    network.setVerbose(0);
    
    //  ----- RUNNING THE NETWORK -----
    network.run(&trainingData, 1, &testData);

    //  ----- EXITING APPLICATION -----
    return 0;
}
