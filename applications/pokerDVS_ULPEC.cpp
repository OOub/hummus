/*
 * pokerDVS_ULPEC.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: Spiking neural network classifying the poker-DVS dataset using STDP for memristor network
 */

#include <iostream>

#include "../source/core.hpp"

#include "../source/randomDistributions/normal.hpp"

#include "../source/learningRules/stdp.hpp"
#include "../source/learningRules/timeInvariantSTDP.hpp"
#include "../source/learningRules/rewardModulatedSTDP.hpp"

#include "../source/neurons/LIF.hpp"
#include "../source/neurons/decisionMaking.hpp"
#include "../source/neurons/parrot.hpp"

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

        /// creating the layers
        auto pixel_grid = network.makeGrid<hummus::Parrot>(32, 32, 1, {}); // input layer
        auto conv_one = network.makeGrid<hummus::LIF>(pixel_grid, 4, 5, 1, {&ti_stdp}, 10, 200, 10, homeostasis, burst); // first convolution
        auto pool_one = network.makeSubsampledGrid<hummus::LIF>(conv_one, {}, 10, 200, 10, false, false); // first pooling
        auto conv_two = network.makeGrid<hummus::LIF>(pool_one, 8, 5, 1, {&ti_stdp}, 10, 1000, 10, homeostasis, burst); // second convolution
        auto pool_two = network.makeSubsampledGrid<hummus::LIF>(conv_two, {}, 10, 200, 10, false, false); // second pooling

        /// connecting the layers
        network.convolution<hummus::Exponential>(pixel_grid, conv_one, 1, hummus::Normal(0.6, 0.1, 0, 0, 0, 1), 100, hummus::synapseType::excitatory);
        network.pooling<hummus::Exponential>(conv_one, pool_one, 1, hummus::Normal(1, 0), 100, hummus::synapseType::excitatory);
        network.convolution<hummus::Exponential>(pool_one, conv_two, 1, hummus::Normal(0.6, 0.1, 0, 0, 0, 1), 100, hummus::synapseType::excitatory);
        network.pooling<hummus::Exponential>(conv_two, pool_two, 1, hummus::Normal(1, 0), 100, hummus::synapseType::excitatory);

        // lateral inhibition
        network.lateralInhibition<hummus::Exponential>(conv_one, 1, hummus::Normal(-1, 0), 100);
        network.lateralInhibition<hummus::Exponential>(pool_one, 1, hummus::Normal(-1, 0), 100);
        network.lateralInhibition<hummus::Exponential>(conv_two, 1, hummus::Normal(-1, 0), 100);
        network.lateralInhibition<hummus::Exponential>(pool_two, 1, hummus::Normal(-1, 0), 100);

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
        bool burst = false;

        /// creating the layers
        auto pixel_grid = network.makeGrid<hummus::Parrot>(32, 32, 1, {}); // input layer
        auto output = network.makeLayer<hummus::LIF>(100, {&ti_stdp}, homeostasis, 200, 10, 10, burst); // output layer with STDP

        /// connecting the layers
        network.allToAll<hummus::Pulse>(pixel_grid, output, 1, hummus::Normal(0.6, 0.1, 0, 0, 0, 1), 100, hummus::synapseType::excitatory);
        network.lateralInhibition<hummus::Pulse>(output, 1, hummus::Normal(-1, 0), 100, hummus::synapseType::excitatory);

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
