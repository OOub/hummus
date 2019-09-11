/*
 * decision_making_test.cpp
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
#include "../source/learningRules/stdp.hpp"
#include "../source/learningRules/myelinPlasticity.hpp"
#include "../source/neurons/parrot.hpp"
#include "../source/neurons/LIF.hpp"
#include "../source/neurons/decisionMaking.hpp"

int main(int argc, char** argv) {

    /// initialisation
    hummus::Network network;
    auto& mp = network.make_addon<hummus::MyelinPlasticity>();
    auto& results = network.make_addon<hummus::Analysis>("../../data/nmnist_testLabel.txt");
    
    /// creating the layers
    auto pixel_grid = network.make_grid<hummus::LIF>(28, 28, 1, {} , 3, 200, 10, false, false); // input layer
    auto hidden_layer = network.make_layer<hummus::LIF>(10, {&mp}, 3, 200, 10, false, false); // hidden layer
    auto decision_layer = network.make_decision<hummus::DecisionMaking>("../../data/nmnist_trainingLabel.txt", 10, 0.6, 2000, {}); // classification layer
    
    /// connecting the layers
    network.all_to_all<hummus::Exponential>(pixel_grid, hidden_layer, 1, hummus::Normal(0.08, 0.02, 10, 3), 60); // all-to-all connection from the pixel grid to the hidden layer
    network.lateral_inhibition<hummus::Exponential>(hidden_layer, 1, hummus::Normal(-1, 0, 0, 1), 100, 60); // lateral inhibition within neurons in the hidden layer
    
    /// Reading data
    hummus::DataParser dataParser;
    auto trainingData = dataParser.read_txt_data("../../data/nmnist_training.txt");
    auto testData = dataParser.read_txt_data("../../data/nmnist_test.txt", 1000);
    
    /// Running the network
    network.verbosity(2);
    network.run_data(trainingData, 0.5, testData);
    
    /// Measuring Classification Accuracy
    results.accuracy();
    
    //  ----- EXITING APPLICATION -----
    return 0;
}
