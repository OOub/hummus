/*
 * pokerDVS.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 22/07/2019
 *
 * Information: Spiking neural network classifying the poker-DVS dataset
 */

#include <iostream>

#include "../source/core.hpp"

#include "../source/randomDistributions/normal.hpp"

#include "../source/learningRules/myelinPlasticity.hpp"

#include "../source/GUI/qt/qtDisplay.hpp"

#include "../source/neurons/LIF.hpp"
#include "../source/neurons/decisionMaking.hpp"
#include "../source/neurons/parrot.hpp"

#include "../source/addons/spikeLogger.hpp"
#include "../source/addons/weightMaps.hpp"
#include "../source/addons/potentialLogger.hpp"
#include "../source/addons/classificationLogger.hpp"
#include "../source/synapses/pulse.hpp"

int main(int argc, char** argv) {
        
    /// Initialisation
    hummus::Network network;
    
    auto& display = network.makeGUI<hummus::QtDisplay>();
    auto& mp = network.makeAddon<hummus::MyelinPlasticity>();
    network.verbosity(0);
    
    /// parameters
    bool homeostasis = false;
    bool burst = false;
    
    /// creating the layers
    auto pixel_grid = network.makeGrid<hummus::LIF>(32, 32, 1, {}, false, 200, 10, 900, false); // input layer
    auto output = network.makeLayer<hummus::LIF>(4, {&mp}, homeostasis, 200, 10, 900, burst); // output layer
    
    /// connecting the layers
    network.allToAll<hummus::Exponential>(pixel_grid, output, 1, hummus::Normal(0.08, 0.02, 5, 3), 100, hummus::synapseType::excitatory);
    network.lateralInhibition<hummus::Exponential>(output, 1, hummus::Normal(-1, 0), 100);
    
    /// Reading data
    hummus::DataParser dataParser;
    auto trainingData = dataParser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/4pips_100rep/DHtraining.txt");
    auto testData = dataParser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/poker-DVS/4pips_100rep/DHtest.txt");
    
    /// Running the network - Learning Phase
    
    auto& mpLog = network.makeAddon<hummus::MyelinPlasticityLogger>("mpLog.bin");
    mpLog.activate_for(output.neurons);
    
    display.setTimeWindow(10000);
    
    network.run(&trainingData, 0.1);
    
    /// Running the network - Test Phase
    network.turnOffLearning();

    network.makeAddon<hummus::SpikeLogger>("test_spikeLog.bin");
    auto& classif = network.makeAddon<hummus::ClassificationLogger>("test_classificationLog.bin");
    classif.activate_for(output.neurons);
    
    network.run(&testData, 1);

    //  ----- EXITING APPLICATION -----
    return 0;
}
