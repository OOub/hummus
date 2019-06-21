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
#include "../source/neurons/parrot.hpp"
#include "../source/GUI/qt/qtDisplay.hpp"
#include "../source/synapses/pulse.hpp"
#include "../source/addons/potentialLogger.hpp"
#include "../source/randomDistributions/normal.hpp"
#include "../source/addons/myelinPlasticityLogger.hpp"
#include "../source/learningRules/myelinPlasticity.hpp"

int main(int argc, char** argv) {
    
    /// ----- PARAMETERS -----
    float direction_conductance         = 200;
    float direction_leakage_conductance = 10;
    float direction_trace_time_constant = 20;
    bool  direction_burst               = false;
    bool  direction_homeostasis         = true;
    
    /// ----- INITIALISATION -----
    
    // initialising the network
    hummus::Network network;
    
    // initialising the loggers
    auto& mp_log = network.makeAddon<hummus::MyelinPlasticityLogger>("localisation_learning.bin");
    auto& potential_log = network.makeAddon<hummus::PotentialLogger>("localisation_potential.bin");
    
    // delay learning rule
//    auto& mp = network.makeAddon<hummus::MyelinPlasticity>(10, 0.1);

    // input layer with 8 channels for each sensor
    auto input = network.makeCircle<hummus::Parrot>(8, {0.3}, {});
    
    /// ----- DIRECTION LAYER -----
    
    // layer that learns the delays
    auto direction = network.makeLayer<hummus::LIF>(100, {}, direction_homeostasis, direction_conductance, direction_leakage_conductance, 0, direction_burst, direction_trace_time_constant);
    
    // connecting input layer with the direction neurons
    network.allToAll<hummus::Exponential>(input, direction, 1, hummus::Normal(1./8, 0, 5, 3, 0, 1, 0, INFINITY), 100); // fixed weight on [0,1], random delays on [0, inf]
    network.lateralInhibition<hummus::Exponential>(direction, 1, hummus::Normal(-1./2, 0), 100);

    // neuron mask for loggers
    mp_log.activate_for(direction.neurons[0]);
    potential_log.activate_for(direction.neurons[0]);
    
    /// ----- DISTANCE LAYER -----
    
    // distance neuron
//    auto distance = network.makeCircle<hummus::LIF>(8, {0.3}, {});
    
    // connecting input layer with the distance neurons
    
    /// ----- USER INTERFACE SETTINGS -----
    
    // initialising the GUI
    auto& display = network.makeGUI<hummus::QtDisplay>();
    
    // settings
    display.setTimeWindow(10000);
    display.trackNeuron(direction.neurons[0]);
    display.plotCurrents(false);
    
    /// ----- RUNNING CALIBRATION -----
    
    // reading the calibration data
    hummus::DataParser parser;
    auto calibration = parser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/localisation/calibration_direction_only_100.txt", false);
    
    // run calibration
    network.verbosity(0);
    network.run(&calibration, 0.1);

    // assigning labels to direction neurons
//    
//    // run test
//    auto test = parser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/localisation/test.txt");
//    network.turnOffLearning();
//    network.run(&test, 0.1);
    
    return 0;
}
