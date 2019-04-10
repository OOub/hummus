/*
 * pokerDVS.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last 9Version: 14/01/2019
 *
 * Information: Spiking neural network classifying the poker-DVS dataset
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/randomDistributions/normal.hpp"

#include "../source/learningRules/timeInvariantSTDP.hpp"

#include "../source/neurons/LIF.hpp"
#include "../source/neurons/input.hpp"

#include "../source/addOns/spikeLogger.hpp"
#include "../source/addOns/weightMaps.hpp"
#include "../source/addOns/potentialLogger.hpp"
#include "../source/addOns/classificationLogger.hpp"
#include "../source/synapticKernels/step.hpp"

int main(int argc, char** argv) {
    
    bool networkType = 1; // choose between feedforward or deep spiking neural network
    
    if (networkType == 1) {
        //  ----- DEEP SPIKING NEURAL NETWORK -----
        
        /// Initialisation
        hummus::PotentialLogger pLog("deepPLog.bin");
        hummus::ClassificationLogger cLog("deepCLog.bin");
        hummus::WeightMaps weightMap1("weightMapsCONV1.bin", "/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtrainingLabel.txt", "/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtestLabel.txt");
        hummus::WeightMaps weightMap2("weightMapsCONV2.bin", "/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtrainingLabel.txt", "/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtestLabel.txt");
        hummus::Network network({&pLog, &cLog, &weightMap1, &weightMap2});
        
        auto ti_stdp = network.makeLearningRule<hummus::TimeInvariantSTDP>(); // time-invariant STDP learning rule
        auto step = network.makeSynapticKernel<hummus::Step>(5); // step synaptic kernel
        
        network.setVerbose(0);
        
        /// parameters
        bool burst = true;
        bool homeostasis = true;
        bool conv_wta = true;
        bool pool_wta = false;

        /// creating the layers
        network.add2dLayer<hummus::Input>(40, 40, 1, {}, nullptr); // input layer
        network.addConvolutionalLayer<hummus::LIF>(network.getLayers()[0], 5, 1, hummus::Normal(0.6, 0.1), 100, 4, {&ti_stdp}, &step, homeostasis, 20, 10, conv_wta, burst); // first convolution
        network.addPoolingLayer<hummus::LIF>(network.getLayers()[1], hummus::Normal(1, 0), 100, {}, &step, false, 20, 10, pool_wta, false); // first pooling
        network.addConvolutionalLayer<hummus::LIF>(network.getLayers()[0], 5, 1, hummus::Normal(0.6, 0.1), 100, 8, {&ti_stdp}, &step, homeostasis, 100, 10, conv_wta, burst); // second convolution
        network.addPoolingLayer<hummus::LIF>(network.getLayers()[1], hummus::Normal(1, 0), 100, {}, &step, false, 20, 10, pool_wta, false); // second pooling
        network.addLayer<hummus::LIF>(2, {&ti_stdp}, &step, homeostasis, 200, 10, conv_wta, burst, 20, 0, 20, 0.1, -50, -70, 100); // output layer with 2 neurons
        
        /// connecting the layers
        network.allToAll(network.getLayers()[4], network.getLayers()[5], hummus::Normal(0.6, 0.1));
        
        pLog.neuronSelection(network.getLayers()[5]);
        weightMap1.neuronSelection(network.getLayers()[1]);
        weightMap2.neuronSelection(network.getLayers()[3]);
        
        /// Reading data
        hummus::DataParser dataParser;
        auto trainingData = dataParser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtraining.txt");
        auto testData = dataParser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtest.txt");
        
        /// Running the network
        network.run(&testData, 0, &testData);
        
    } else if (networkType == 0){
        // ----- SIMPLE FEEDFORWARD -----
        
        /// Initialisation
        hummus::PotentialLogger pLog("simplePLog.bin");
        hummus::ClassificationLogger cLog("simpleCLog.bin");
        hummus::Network network({&pLog, &cLog});
        
        auto ti_stdp = network.makeLearningRule<hummus::TimeInvariantSTDP>(); // time-invariant STDP learning rule
        auto step = network.makeSynapticKernel<hummus::Step>(5); // step synaptic kernel
        
        /// parameters
        bool homeostasis = true;
        bool wta = true;
        bool burst = true;
        
        /// creating the layers
        network.add2dLayer<hummus::Input>(34, 34, 1, {}, nullptr); // input layer
        network.addLayer<hummus::LIF>(5000, {&ti_stdp}, &step, homeostasis, 20, 10, wta, burst, 20, 0, 40, 1, -50, -70, 100); // hidden layer with STDP
        network.addLayer<hummus::LIF>(2, {&ti_stdp}, &step, homeostasis, 500, 10, wta, burst, 20, 0, 40, 1, -60, -70, 100); // output layer with 2 neurons

        /// connecting the layers
        network.allToAll(network.getLayers()[0], network.getLayers()[1], hummus::Normal(0.8, 0.1));
        network.allToAll(network.getLayers()[1], network.getLayers()[2], hummus::Normal(0.8, 0.1));
        
        pLog.neuronSelection(network.getLayers()[2]);
        
        /// Reading data
        hummus::DataParser dataParser;
        auto trainingData = dataParser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtraining.txt");
        auto testData = dataParser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtest.txt");
        
        /// Running the network
        network.run(&trainingData, 0, &testData);
    }

    //  ----- EXITING APPLICATION -----
    return 0;
}
