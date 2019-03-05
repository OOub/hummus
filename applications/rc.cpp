/*
 * rc.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 05/03/2019
 *
 * Information: Reservoir network for N-MNIST without a readout function
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/rand.hpp"
#include "../source/dataParser.hpp"
#include "../source/neurons/inputNeuron.hpp"
#include "../source/neurons/LIF.hpp"
#include "../source/neurons/IF.hpp"
#include "../source/addOns/spikeLogger.hpp"
#include "../source/addOns/potentialLogger.hpp"

#include "tbb/tbb.h"

int main(int argc, char** argv) {

    // ----- RESERVOIR PARAMETERS -----
    int numberOfNeurons = 10;
    float weightMean = 1; // gaussian parameter - for weights
    float weightStdDev = 1; // standard deviation for gaussian - for weights
    int feedforwardProbability = 100; // percentage likelihood of feedforward connections
    int feedbackProbability = 100; // percentage likelihood of feedback connections
    int selfExcitationProbability = 100; // percentage likelihood of self-excitation
    float resetCurrent = 10; // current step function reset value (integration time)
    float decayPotential = 20; // time constant for membrane potential (decay)
    int refractoryPeriod = 3; // neuron inactive for specified time after each spike
    bool wta = false; // winner-takes-all algorithm

    // ----- IMPORTING DATA -----
    hummus::DataParser parser;
    auto data = parser.readData("path to file");

    //  ----- INITIALISING THE NETWORK -----
    hummus::SpikeLogger spikeLog("rcSpike.bin");
    hummus::PotentialLogger potentialLog("rervoirPotential.bin");
    hummus::Network network({&spikeLog, &potentialLog});

    //  ----- CREATING THE NETWORK -----

    // pixel grid layer
    network.add2dLayer<hummus::InputNeuron>(28, 28, 1, {});

    // reservoir layer
    network.addReservoir<hummus::LIF>(numberOfNeurons, weightMean, weightStdDev, feedforwardProbability, feedbackProbability, selfExcitationProbability, false, false, resetCurrent, decayPotential, refractoryPeriod, wta);

    // initialising the potentialLoggers
    potentialLog.neuronSelection(network.getLayers()[1]);

    //  ----- RUNNING THE NETWORK ASYNCHRONOUSLY-----
    network.run(&data, 0);

    //  ----- EXITING APPLICATION -----
    return 0;
}
