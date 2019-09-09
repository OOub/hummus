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
#include "../source/neurons/LIF.hpp"
#include "../source/neurons/parrot.hpp"
#include "../source/GUI/display.hpp"
#include "../source/addons/potentialLogger.hpp"
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
    auto& mp_log = network.make_addon<hummus::MyelinPlasticityLogger>("localisation_learning.bin");
    auto& potential_log = network.make_addon<hummus::PotentialLogger>("localisation_potential.bin");

    // delay learning rule
    auto& mp = network.make_addon<hummus::MyelinPlasticity>();

    // input layer with 8 channels for each sensor
    auto input = network.make_circle<hummus::Parrot>(8, {0.3}, {});

    /// ----- DIRECTION LAYER -----

    // layer that learns the delays
    auto direction = network.make_layer<hummus::LIF>(16, {&mp}, 0, direction_conductance, direction_leakage_conductance, direction_homeostasis, direction_burst, direction_trace_time_constant);

    // connecting input layer with the direction neurons
    network.all_to_all<hummus::Exponential>(input, direction, 1, hummus::Normal(1./8, 0, 5, 3, 0, 1, 0, INFINITY), 100); // fixed weight on [0,1], random delays on [0, inf]
    network.lateral_inhibition<hummus::Exponential>(direction, 1, hummus::Normal(-1, 0, 0, 1), 100);

    // neuron mask for loggers
    mp_log.activate_for(direction.neurons[0]);
    potential_log.activate_for(direction.neurons[0]);

    /// ----- DISTANCE LAYER -----

    // distance neuron
//    auto distance = network.make_circle<hummus::LIF>(8, {0.3}, {});

    // connecting input layer with the distance neurons

    /// ----- USER INTERFACE SETTINGS -----

    // initialising the GUI
    auto& display = network.make_gui<hummus::Display>();

    // settings
    display.set_time_window(10000);
    display.track_neuron(direction.neurons[0]);
    display.plot_currents(false);

    /// ----- RUNNING CALIBRATION -----

    // reading the calibration data
    hummus::DataParser parser;
    auto calibration = parser.read_txt_data("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/localisation/calibration_direction_only_100.txt", false);

    // run calibration
    network.verbosity(0);
    network.run_data(&calibration, 0.1);

    // assigning labels to direction neurons
//
//    // run test
//    auto test = parser.read_data("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/localisation/test.txt");
//    network.turn_off_learning();
//    network.run_data(&test, 0.1);

    return 0;
}
