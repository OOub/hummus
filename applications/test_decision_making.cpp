/*
 * test_decision_making.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/07/2019
 *
 * Information: Example of the decision-making at work.
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/addons/analysis.hpp"
#include "../source/randomDistributions/normal.hpp"
#include "../source/learningRules/stdp.hpp"
#include "../source/learningRules/myelinPlasticity.hpp"
#include "../source/GUI/qt/qtDisplay.hpp"
#include "../source/neurons/parrot.hpp"
#include "../source/neurons/LIF.hpp"
#include "../source/neurons/decisionMaking.hpp"
#include "../source/synapses/exponential.hpp"

int main(int argc, char** argv) {

    /// initialisation
    hummus::Network network;
    auto& stdp = network.makeAddon<hummus::STDP>();
    auto& results = network.makeAddon<hummus::Analysis>("../../data/nmnist_testLabel.txt");
    
    /// creating the layers
    auto pixel_grid = network.makeGrid<hummus::Parrot>(28, 28, 1, {}); // input layer
    auto hidden_layer = network.makeLayer<hummus::LIF>(100, {&stdp}, 3, 200, 10, false, false); // hidden layer
    auto decision_layer = network.makeDecision<hummus::DecisionMaking>("../../data/nmnist_trainingLabel.txt", 10, 0.6, 2000, {}); // classification layer
    
    /// connecting the layers
    network.allToAll<hummus::Exponential>(pixel_grid, hidden_layer, 1, hummus::Normal(0.08, 0.02, 10, 3), 60, hummus::synapseType::excitatory); // all-to-all connection from the pixel grid to the hidden layer
    network.lateralInhibition<hummus::Exponential>(hidden_layer, 1, hummus::Normal(-1, 0, 0, 1), 100, 100); // lateral inhibition within neurons in the hidden layer
    
    /// Reading data
    hummus::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/nmnist_training.txt");
    auto testData = dataParser.readData("../../data/nmnist_test.txt", 1000);
    
    /// Running the network
    network.verbosity(0);
    network.run(&trainingData, 0.5, &testData);
    
    /// Measuring Classification Accuracy
    results.accuracy();
    
    //  ----- EXITING APPLICATION -----
    return 0;
}
