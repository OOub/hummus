/*
 * localisation.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 01/05/2019
 *
 * Information: Network for wave triangulation from an array of 8 piezoelectric sensors using delay learning
 * grid of 2D neurons, each associated with a specific temporal pattern. the final output neuron give us the position of the touch
 * a calibration step is necessary to find the delays for each grid point
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/dataParser.hpp"
#include "../source/neurons/LIF.hpp"
#include "../source/neurons/input.hpp"
#include "../source/GUI/qt/qtDisplay.hpp"
#include "../source/synapticKernels/step.hpp"
#include "../source/randomDistributions/normal.hpp"
#include "../source/learningRules/myelinPlasticity.hpp"

int main(int argc, char** argv) {
    
    // semi-supervised - choose coordinates according to which neuron responds first - organising the network geometry knowing the expected coordinates.
    
    // initialising the network
    hummus::Network network;
    
    // initialise GUI
    auto& display = network.makeGUI<hummus::QtDisplay>();
    
    // event-based synaptic kernel
    auto& kernel = network.makeSynapticKernel<hummus::Exponential>();

    // learning rule - needs to be modified to adapt to the relative timing between the first two sensors
    auto& mp = network.makeAddon<hummus::MyelinPlasticity>();

    // input layer with 8 channels for each sensor
    auto input = network.makeLayer<hummus::Input>(8, {});

    float potentialDecay = 20;
    float eligibilityDecay = 20;
    bool wta = true;
    bool burst = false;
    bool homeostasis = false;
    
    // myelin plasticity layer that learns the delays
    auto hidden = network.makeLayer<hummus::LIF>(16, {&mp}, &kernel, homeostasis, potentialDecay, 0, wta, burst, eligibilityDecay);
    
    // connecting input layer with the myelin plasticity neurons
    network.allToAll(input, hidden, hummus::Normal(1./8, 0, 3, 1, 0, 1, 0, INFINITY));
    
    // reading the data
    hummus::DataParser parser;

    auto calibration = parser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/localisation/calibration_direction_only_100.txt");

    // display settings
    display.setTimeWindow(10000);
    
    // run calibration process
    network.run(&calibration, 0.1);
    
    // assign a 2D structure to the network according to which neurons learned the different calibration positions
    
    // run test process
//    auto test = parser.readData("/Users/omaroubari/Documents/Education/UPMC - PhD/Datasets/hummus_data/localisation/test.txt");
//    network.turnOffLearning();
//    network.run(&test, 0.1);
    
    return 0;
}
