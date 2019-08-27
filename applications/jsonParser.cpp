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

#include "../source/builder.hpp"
#include "../source/core.hpp"
#include "../source/GUI/display.hpp"

int main(int argc, char** argv) {
    hummus::Network network;
    auto& display = network.makeGUI<hummus::Display>();
    
    hummus::Builder bob(&network);
    bob.import("../../data/testSave.json");
    
    //  ----- INJECTING SPIKES -----
    network.injectSpike(0, 10);
    network.injectSpike(0, 11);
    network.injectSpike(0, 30);
    
    //  ----- DISPLAY SETTINGS -----
    display.setTimeWindow(100);
    display.trackNeuron(1);
    
    network.run(100, 0.1);
    
    return 0;
}
