/*
 * basic_test.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: Example of a basic spiking neural network.
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
    
    //  ----- INITIALISING ADD-ONS -----
    network.make_addon<hummus::SpikeLogger>("spikeLog.bin");

    // ----- INITIALISING GUI -----
    auto& display = network.make_gui<hummus::Display>();
    
    //  ----- CREATING THE NETWORK -----
    // creating layers of neurons
    auto input = network.make_layer<hummus::Parrot>(1, {});
    auto output = network.make_layer<hummus::LIF>(2, {}, 3, 200, 10, false, false);

    //  ----- CONNECTING THE NETWORK -----
    network.all_to_all<hummus::Exponential>(input, output, 1, hummus::Normal(1./2, 0, 1, 0.5), 100);
    network.lateral_inhibition<hummus::Exponential>(output, 1, hummus::Normal(-1, 0, 0, 1), 100);

    //  ----- INJECTING SPIKES -----
    network.inject_spike(0, 10);
    network.inject_spike(0, 12);
    network.inject_spike(0, 30);

    //  ----- DISPLAY SETTINGS -----
    display.set_time_window(100);
    display.track_neuron(1);
    display.plot_currents();

    //  ----- RUNNING THE NETWORK -----
    network.verbosity(1);
    network.run(100, 0.1);

    //  ----- SAVE THE NETWORK IN A JSON FILE -----
    network.save("testSave");

    //  ----- EXITING APPLICATION -----
    return 0;
}
