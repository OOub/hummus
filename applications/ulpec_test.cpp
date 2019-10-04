/*
 * ulpec_test.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 02/10/2019
 *
 * Information: ULPEC demonstrator simulation test
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/GUI/display.hpp"
#include "../source/neurons/pulse_generator.hpp"
#include "../source/neurons/hardware.hpp"
#include "../source/addons/spike_logger.hpp"

int main(int argc, char** argv) {
    /// parameters
    double runtime = 500; // microseconds
    bool use_gui = false;
    
    /// initialisation
    hummus::Network network;
    hummus::DataParser parser;
    network.make_addon<hummus::SpikeLogger>("ulpec_log.bin");
    
    if (use_gui) {
        auto& display = network.make_gui<hummus::Display>();
        display.set_time_window(runtime); // in microseconds
        display.track_neuron(2);
        display.plot_currents();
    }
    
    /// creating the layers
    auto input = network.make_layer<hummus::Pulse_Generator>(2, {});
    auto output = network.make_layer<hummus::Hardware>(1, {});
    
    /// connecting the layers with memristive synapses

    /// injecting artificial spikes
    // 25 spikes over 500 microseconds separated by 10 us for neuron 0
    
    // 15 spikes over 500 microseconds separated by 10 us for neuron 1

    /// running network
    network.verbosity(1);
    network.run(runtime, 0);


    /// Exiting Application
    return 0;
}
