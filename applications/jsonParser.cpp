/*
 * json_parser.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 31/05/2018
 *
 * Information: Example of a network being build from a JSON save file
 */

#include <string>
#include <iostream>

#include "../source/core.hpp"
#include "../source/builder.hpp"
#include "../source/GUI/display.hpp"

int main(int argc, char** argv) {
    hummus::Network network;
    auto& display = network.make_gui<hummus::Display>();
    
    hummus::Builder bob(&network);
    bob.import("../../data/test_save.json");
    
    //  ----- INJECTING SPIKES -----
    network.inject_spike(0, 10);
    network.inject_spike(0, 11);
    network.inject_spike(0, 30);
    
    //  ----- DISPLAY SETTINGS -----
    display.set_time_window(100);
    display.track_neuron(2);
    
    network.run(100, 0.1);
    
    return 0;
}
