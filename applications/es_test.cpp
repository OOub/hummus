/*
 * es_test.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 17/09/2019
 *
 * Information: Example of a spiking neural network running as eventstream file (Sepia C++ library).
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/GUI/display.hpp"
#include "../source/neurons/parrot.hpp"
#include "../source/neurons/decisionMaking.hpp"
#include "../source/neurons/LIF.hpp"
#include "../source/addons/spikeLogger.hpp"
#include "../source/learningRules/myelinPlasticity.hpp"
#include "../source/learningRules/stdp.hpp"
#include "../source/dataParser.hpp"

int main(int argc, char** argv) {
    //  ----- INITIALISING THE NETWORK -----
    hummus::Network network;

    // ----- INITIALISING GUI -----
    auto& display = network.make_gui<hummus::Display>();
    
    //  ----- CREATING THE NETWORK -----
    auto input = network.make_grid<hummus::Parrot>(34, 34, 1, {});
    auto output = network.make_layer<hummus::LIF>(2, {}, 3, 200, 10, false, false);

    //  ----- CONNECTING THE NETWORK -----
    network.all_to_all<hummus::Square>(input, output, 1, hummus::Normal(0.5f, 0, 1, 0.5f), 100);
    network.lateral_inhibition<hummus::Square>(output, 1, hummus::Normal(-1, 0, 0, 1), 100);

    //  ----- DISPLAY SETTINGS -----
    display.set_time_window(100000);
    display.track_neuron(1);
    display.plot_currents();

    //  ----- RUNNING THE NETWORK -----
    network.verbosity(1);
    network.run_es("../../data/00002.es", false, 100000);

    //  ----- EXITING APPLICATION -----
    return 0;
}
