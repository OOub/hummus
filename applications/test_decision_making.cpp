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

#include "../source/randomDistributions/normal.hpp"

#include "../source/learningRules/myelinPlasticity.hpp"

#include "../source/GUI/qt/qtDisplay.hpp"

#include "../source/neurons/parrot.hpp"
#include "../source/neurons/LIF.hpp"
#include "../source/neurons/decisionMaking.hpp"

#include "../source/synapses/exponential.hpp"

int main(int argc, char** argv) {

    /// initialisation
    hummus::Network network;
    auto& display = network.makeGUI<hummus::QtDisplay>();
    auto& mp = network.makeAddon<hummus::MyelinPlasticity>();
    
    /// creating the layers
    auto pixel_grid = network.makeGrid<hummus::LIF>(32, 32, 1, {}, false, 200, 10, 0, false); // input layer
    auto hidden_layer = network.makeLayer<hummus::LIF>(100, {&mp}, false, 200, 10, 900, false); // hidden layer
    auto decision_layer = network.makeDecision<hummus::DecisionMaking>("../../data/trainingDecisionLabel.txt", 10, 0.6, 1000, {}); // classification layer
    
    /// connecting the layers
    network.allToAll<hummus::Exponential>(pixel_grid, hidden_layer, 1, hummus::Normal(0.08, 0.02, 5, 3), 80, hummus::synapseType::excitatory);
    network.allToAll<hummus::Exponential>(hidden_layer, decision_layer, 1, hummus::Normal(1, 0), 100, hummus::synapseType::excitatory);
    
    network.lateralInhibition<hummus::Exponential>(hidden_layer, 1, hummus::Normal(-1, 0), 20);
	network.lateralInhibition<hummus::Exponential>(decision_layer, 1, hummus::Normal(-1, 0), 100);
    
    /// Reading data
    hummus::DataParser dataParser;
    auto trainingData = dataParser.readData("../../data/trainingDecision.txt");
    auto testData = dataParser.readData("../../data/testDecision.txt");
    
    /// Display settings
    display.setTimeWindow(10000);
    
    /// Running the network
    network.run(&trainingData, 1, &testData);
    
    //  ----- EXITING APPLICATION -----
    return 0;
}
