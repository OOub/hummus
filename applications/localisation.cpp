/*
 * localisation.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 03/05/2019
 *
 * Information: Network for wave triangulation from an array of 8 piezoelectric sensors using delay learning
 * grid of 2D neurons, each associated with a specific temporal pattern. the final output neuron give us the position of the touch
 * a calibration step is necessary to find the delays for each grid point
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/dataParser.hpp"
#include "../source/neurons/LIF.hpp"
#include "../source/neurons/input.hpp"
#include "../source/GUI/qt/qtDisplay.hpp"
#include "../source/synapses/pulse.hpp"
#include "../source/addons/potentialLogger.hpp"
#include "../source/randomDistributions/normal.hpp"
#include "../source/addons/myelinPlasticityLogger.hpp"
#include "../source/learningRules/myelinPlasticity.hpp"

int main(int argc, char** argv) {
    
    /// ----- PARAMETERS -----
    float direction_potentialDecay   = 20;
    float direction_currentDecay     = 10;
    float direction_eligibilityDecay = 20;
    bool  direction_wta              = true;
    bool  direction_burst            = false;
    bool  direction_homeostasis      = true;
    
    /// ----- INITIALISATION -----
    
    // initialising the network
    hummus::Network network;
    
    // initialising the loggers
    auto& mp_log = network.makeAddon<hummus::MyelinPlasticityLogger>("localisation_learning.bin");
    auto& potential_log = network.makeAddon<hummus::PotentialLogger>("localisation_potential.bin");
    
    // delay learning rule
    auto& mp = network.makeAddon<hummus::MyelinPlasticity>(1, 1, 1, 0.1);

    // input layer with 8 channels for each sensor
    auto input = network.makeCircle<hummus::Input>(8, {0.3}, {});
    
    /// ----- DIRECTION LAYER -----
    
    // layer that learns the delays
    auto direction = network.makeLayer<hummus::LIF>(50, {&mp}, direction_homeostasis, direction_potentialDecay, direction_currentDecay, 0, direction_wta, direction_burst, direction_eligibilityDecay);
    
    // connecting input layer with the direction neurons
    network.allToAll<hummus::Exponential>(input, direction, 1, hummus::Normal(1./8, 0, 5, 3, 0, 1, 0, INFINITY), 100); // fixed weight on [0,1], random delays on [0, inf]
    
    // neuron mask for loggers
    mp_log.activate_for(direction.neurons[0]);
    potential_log.activate_for(direction.neurons[0]);
    
    /// ----- DISTANCE LAYER -----
    
    // distance neuron
    auto distance = network.makeCircle<hummus::LIF>(8, {0.3}, {});
    
    // connecting input layer with the distance neurons
    
    /// ----- USER INTERFACE SETTINGS -----
    
    // initialising the GUI
    auto& display = network.makeGUI<hummus::QtDisplay>();
    
    // settings
    display.setTimeWindow(10000);
    display.trackNeuron(direction.neurons[0]);
    
    /// ----- RUNNING CALIBRATION -----
    
    // reading the training data
    hummus::DataParser parser;
    auto calibration = parser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/localisation/calibration_direction_only_100.txt", false);
    
    // run calibration
    network.verbosity(1);
    network.run(&calibration, 0.1);

    // assigning labels to direction neurons
    
    // run test
    auto test = parser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/localisation/test.txt");
    network.turnOffLearning();
    network.run(&test, 0.1);
    
    return 0;
}
