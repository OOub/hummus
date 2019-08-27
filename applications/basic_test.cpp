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
    
    hummus::DataParser parser;
    auto train_data = parser.importNmnist("/Users/omaroubari/Downloads/N-MNIST/Train", 100);
    
    //  ----- INITIALISING THE NETWORK -----
    hummus::Network network;
    
    //  ----- INITIALISING ADD-ONS -----
    network.makeAddon<hummus::SpikeLogger>("spikeLog.bin");

    // ----- INITIALISING GUI -----
    auto& display = network.makeGUI<hummus::Display>();
    
    //  ----- CREATING THE NETWORK -----
    // creating layers of neurons
    auto input = network.makeLayer<hummus::Parrot>(1, {});
    auto output = network.makeLayer<hummus::LIF>(2, {}, 3, 200, 10, false, false);

    //  ----- CONNECTING THE NETWORK -----
    network.allToAll<hummus::Exponential>(input, output, 1, hummus::Normal(1./2, 0, 1, 0.5), 100);
    network.lateralInhibition<hummus::Exponential>(output, 1, hummus::Normal(-1, 0, 0, 1), 100);

    //  ----- INJECTING SPIKES -----
    network.injectSpike(0, 10);
    network.injectSpike(0, 12);
    network.injectSpike(0, 30);

    //  ----- DISPLAY SETTINGS -----
    display.setTimeWindow(100);
    display.trackNeuron(1);
    display.plotCurrents();

    //  ----- RUNNING THE NETWORK -----
    network.verbosity(1);
    network.run(100, 0.1);

    //  ----- SAVE THE NETWORK IN A JSON FILE -----
    network.save("testSave");

    //  ----- EXITING APPLICATION -----
    return 0;
}
