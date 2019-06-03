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

#include "../source/learningRules/stdp.hpp"
#include "../source/learningRules/timeInvariantSTDP.hpp"
#include "../source/learningRules/rewardModulatedSTDP.hpp"

#include "../source/neurons/LIF.hpp"
#include "../source/neurons/decisionMaking.hpp"
#include "../source/neurons/input.hpp"

#include "../source/addons/spikeLogger.hpp"
#include "../source/addons/weightMaps.hpp"
#include "../source/addons/potentialLogger.hpp"
#include "../source/addons/classificationLogger.hpp"
#include "../source/synapticKernels/step.hpp"

int main(int argc, char** argv) {
    
    bool deepNetwork = true; // choose between feedforward or deep spiking neural network
    
    if (deepNetwork) {
        //  ----- DEEP SPIKING NEURAL NETWORK -----
        
        /// Initialisation
        hummus::Network network;
        auto& pLog = network.makeAddon<hummus::PotentialLogger>("deepPLog.bin");
        network.makeAddon<hummus::ClassificationLogger>("deepCLog.bin");
        auto& weightMap1 = network.makeAddon<hummus::WeightMaps>("weightMapsCONV1.bin", "/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtrainingLabel.txt", "/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtestLabel.txt");
        auto& weightMap2 = network.makeAddon<hummus::WeightMaps>("weightMapsCONV2.bin", "/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtrainingLabel.txt", "/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtestLabel.txt");
        
        auto& ti_stdp = network.makeAddon<hummus::TimeInvariantSTDP>(); // time-invariant STDP learning rule
        auto& step = network.makeSynapticKernel<hummus::Step>(5); // step synaptic kernel
        
        network.verbosity(0);
        
        /// parameters
        bool burst = false;
        bool homeostasis = true;
        bool conv_wta = true;
        bool pool_wta = false;

        /// creating the layers
        network.make2dLayer<hummus::Input>(32, 32, 1, {}, nullptr); // input layer
        network.makeConvolutionalLayer<hummus::LIF>(network.getLayers()[0], 5, 1, hummus::Normal(0.6, 0.1, 0, 0, 0, 1), 100, 4, {&ti_stdp}, &step, homeostasis, 20, 10, conv_wta, burst); // first convolution
        network.makePoolingLayer<hummus::LIF>(network.getLayers()[1], hummus::Normal(1, 0), 100, {}, &step, false, 20, 10, pool_wta, false); // first pooling
        network.makeConvolutionalLayer<hummus::LIF>(network.getLayers()[2], 5, 1, hummus::Normal(0.6, 0.1, 0, 0, 0, 1), 100, 8, {&ti_stdp}, &step, homeostasis, 100, 10, conv_wta, burst); // second convolution
        network.makePoolingLayer<hummus::LIF>(network.getLayers()[3], hummus::Normal(1, 0), 100, {}, &step, false, 20, 10, pool_wta, false); // second pooling
//        network.makeLayer<hummus::LIF>(<#int _numberOfNeurons#>, <#std::vector<Addon *> _addons#>, <#Args &&args...#>)
        
        /// connecting the layers
        network.allToAll(network.getLayers()[4], network.getLayers()[5], hummus::Normal(0.6, 0.1, 0, 0, 0, 1));
        
        pLog.activate_for(network.getLayers()[5].neurons);
        weightMap1.activate_for(network.getLayers()[1].neurons);
        weightMap2.activate_for(network.getLayers()[3].neurons);
        
        /// Reading data
        hummus::DataParser dataParser;
        auto trainingData = dataParser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtraining.txt");
        auto testData = dataParser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtest.txt");
        
        /// Running the network
        network.run(&trainingData, 0);
        
    } else {

        // ----- SIMPLE FEEDFORWARD -----
        
        /// Initialisation
        hummus::Network network;
        
        auto& ti_stdp = network.makeAddon<hummus::TimeInvariantSTDP>(); // time-invariant STDP learning rule
        auto& step = network.makeSynapticKernel<hummus::Step>(5); // step synaptic kernel
        
        network.verbosity(0);
        
        /// parameters
        bool homeostasis = true;
        bool wta = true;
        bool burst = false;
        
        /// creating the layers
        auto pixel_grid = network.make2dLayer<hummus::Input>(32, 32, 1, {}); // input layer
        auto output = network.makeLayer<hummus::LIF>(100, {&ti_stdp}, &step, homeostasis, 20, 10, wta, burst, 20, 0, 40, 1, -50, -70, 100); // output layer with STDP

        /// connecting the layers
        network.allToAll(pixel_grid, output, hummus::Normal(0.6, 0.1, 0, 0, 0, 1));
        
        /// Reading data
        hummus::DataParser dataParser;
        auto trainingData = dataParser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtraining.txt");
        auto testData = dataParser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtest.txt");
        
        /// Running the network - Learning Phase
        network.run(&trainingData, 0);
        
        /// Re-Running the network - Training Data Collection
        network.turnOffLearning();

        auto& simpleTrainingPLog = network.makeAddon<hummus::PotentialLogger>("simpleTrainingPLog.bin");
        simpleTrainingPLog.activate_for(output.neurons);

        network.run(&trainingData, 0);
        
        /// Re-Running the network - Test Phase
        network.turnOffLearning();

        auto& simpleTestPLog = network.makeAddon<hummus::PotentialLogger>("simpleTestPLog.bin");
        simpleTestPLog.activate_for(output.neurons);

        network.run(&testData, 0);
        
    }

    //  ----- EXITING APPLICATION -----
    return 0;
}
