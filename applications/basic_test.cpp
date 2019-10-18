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
#include "../source/neurons/cuba_lif.hpp"
#include "../source/addons/spike_logger.hpp"

int main(int argc, char** argv) {
    
    //  ----- INITIALISING THE NETWORK -----
    hummus::Network network;

    //  ----- INITIALISING ADD-ONS -----
    network.make_addon<hummus::SpikeLogger>("spike_log.bin");

    // ----- INITIALISING GUI -----
    auto& display = network.make_gui<hummus::Display>();
    
    //  ----- CREATING THE NETWORK -----
    auto input = network.make_layer<hummus::Parrot>(1, {});
    auto output = network.make_layer<hummus::CUBA_LIF>(2, {}, 3, 200, 10, false, false, false);
    
    //  ----- CONNECTING THE NETWORK -----
    network.all_to_all<hummus::Square>(input, output, 1, hummus::Normal(0.5, 0, 0, 1), 100);
    network.lateral_inhibition<hummus::Square>(output, 1, hummus::Normal(-1, 0, 0, 0), 100);

    //  ----- INJECTING SPIKES -----
    network.inject_spike(0, 10);
    network.inject_spike(0, 12);
    network.inject_spike(0, 30);

    //  ----- DISPLAY SETTINGS -----
//    display.set_time_window(100);
//    display.track_neuron(1);
//    display.plot_currents();

    //  ----- RUNNING THE NETWORK -----
    network.verbosity(1);
    network.run(100, 0.1);

    //  ----- SAVE THE NETWORK IN A JSON FILE -----
    network.save("test_save");

    //  ----- EXITING APPLICATION -----
    return 0;
}
