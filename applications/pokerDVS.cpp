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
#include "../source/synapses/pulse.hpp"

int main(int argc, char** argv) {
    
    bool deepNetwork = false; // choose between feedforward or deep spiking neural network
    
    if (deepNetwork) {
        //  ----- DEEP SPIKING NEURAL NETWORK -----
        
        /// Initialisation
        hummus::Network network;
        auto& pLog = network.makeAddon<hummus::PotentialLogger>("deepPLog.bin");
        network.makeAddon<hummus::ClassificationLogger>("deepCLog.bin");
        auto& weightMap1 = network.makeAddon<hummus::WeightMaps>("weightMapsCONV1.bin", "/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtrainingLabel.txt", "/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtestLabel.txt");
        auto& weightMap2 = network.makeAddon<hummus::WeightMaps>("weightMapsCONV2.bin", "/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtrainingLabel.txt", "/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/DHtestLabel.txt");
        
        auto& ti_stdp = network.makeAddon<hummus::TimeInvariantSTDP>(); // time-invariant STDP learning rule
        
        network.verbosity(0);
        
        /// parameters
        bool burst = false;
        bool homeostasis = true;
        bool conv_wta = true;
        bool pool_wta = false;

        /// creating the layers
        auto pixel_grid = network.makeGrid<hummus::Input>(32, 32, 1, {}); // input layer
        auto conv_one = network.makeGrid<hummus::LIF>(pixel_grid, 4, 5, 1, {&ti_stdp}, homeostasis, 20, 10, 10, conv_wta, burst); // first convolution
        auto pool_one = network.makeSubsampledGrid<hummus::LIF>(conv_one, {}, false, 20, 10, 10, pool_wta, false); // first pooling
        auto conv_two = network.makeGrid<hummus::LIF>(pool_one, 8, 5, 1, {&ti_stdp}, homeostasis, 100, 50, 10, conv_wta, burst); // second convolution
        auto pool_two = network.makeSubsampledGrid<hummus::LIF>(conv_two, {}, false, 20, 10, 10, pool_wta, false); // second pooling
        
        /// connecting the layers
        
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
        
        network.verbosity(0);
        
        /// parameters
        bool homeostasis = true;
        bool wta = true;
        bool burst = false;
        
        /// creating the layers
        auto pixel_grid = network.makeGrid<hummus::Input>(32, 32, 1, {}); // input layer
        auto output = network.makeLayer<hummus::LIF>(100, {&ti_stdp}, homeostasis, 20, 10, 10, wta, burst); // output layer with STDP

        //float _eligibilityDecay=20, float _decayHomeostasis=20, float _homeostasisBeta=0.1, float _threshold=-50, float _restingPotential=-70
        
        /// connecting the layers
        network.allToAll<hummus::Pulse>(pixel_grid, output, hummus::Normal(0.6, 0.1, 0, 0, 0, 1), 100);
        
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
