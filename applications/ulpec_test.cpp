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
#include "../source/neurons/ulpec_input.hpp"
#include "../source/neurons/ulpec_lif.hpp"
#include "../source/addons/potential_logger.hpp"

int main(int argc, char** argv) {
    /// parameters
    double runtime = 500; // microseconds
    bool use_gui = true;
    bool plot_currents = false;
    

    /// initialisation
    hummus::Network network;
    hummus::DataParser parser;
    auto& potential_logger = network.make_addon<hummus::PotentialLogger>("ulpec_v_log.bin");
    potential_logger.activate_for(2);
    
    if (use_gui) {
        auto& display = network.make_gui<hummus::Display>();
        display.set_time_window(runtime); // in microseconds
        display.set_potential_limits(0, 1.5);
        display.track_neuron(2);
        display.hardware_acceleration(false);
        if (plot_currents) {
            display.plot_currents();
            display.set_current_limits(0, 5e-8);
        }
    }
    
    /// creating the layers
    auto input = network.make_layer<hummus::ULPEC_Input>(2, {}, 0, 1.2, 0, 10, 1, true);
    auto output = network.make_layer<hummus::ULPEC_LIF>(1, {}, 10, 5e-12, 0, 0, 12e-9, 0, 650, true, 0.5, 10, 1.5);
    
    /// changing the time_constant of the second input neuron from 10 us to 15us
    network.get_neurons()[1]->set_membrane_time_constant(15);
    
    /// connecting the input and output layer with memristive synapses
    network.all_to_all<hummus::Memristor>(input, output, 1, hummus::Normal(1e-5), 100, false);
    
    // creating feedback connections from the output back to the input
    network.all_to_all(output, input, 1, hummus::Normal(), 100);
    
    /// injecting artificial spikes
    // 25 spikes over 500 microseconds separated by 20 us for neuron 0
    std::vector<hummus::event> pre_one;
    for (int i=0; i<25; i++) {
        pre_one.emplace_back(hummus::event{static_cast<double>(i*20+10), 0});
    }
    network.inject_input(pre_one);
    
    // 20 spikes over 500 microseconds separated by 25 us for neuron 1
    std::vector<hummus::event> pre_two;
    for (auto i=0; i<20; i++) {
        pre_two.emplace_back(hummus::event{static_cast<double>(i*25+10), 1});
    }
    network.inject_input(pre_two);
    
    /// running network
    network.verbosity(1);
    network.run(runtime);

    /// Exiting Application
    return 0;
}
