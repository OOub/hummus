/*
 * rc.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 05/03/2019
 *
 * Information: Reservoir network for N-MNIST without a readout function. Works with command-line arguments.
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

int main(int argc, char** argv) {
    if (argc < 16) {
        std::cout << "REQUIRED ARGUMENTS:\n" << "path to data file \n" << "name of output spike file\n" <<  "name of output potential file\n" << "pixel grid width (int) \n" << "pixel grid height (int)\n" << "number of neurons inside the reservoir (int) \n" << "gaussian mean for weights (float) \n" << "gaussian standard deviation for weights (float)\n" << "percentage likelihood of feedforward connections (int)\n" << "percentage likelihood of feedback connections (int) \n" << "percentage likelihood of self-excitation (int)\n" << "current step function reset value (int)\n" << "potential decay (int)\n" << "refractory period (int)\n" << "winner-takes-all (0 or 1 for true or false)\n" << "threshold adaptation to firing rate (0 or 1 for true or false)" << std::endl;
        
        throw std::runtime_error("not enough arguments");
    }
    
    // ----- RESERVOIR PARAMETERS -----
    std::string dataPath = argv[1]; // path to data file
    std::string spikeLogName = argv[2]; // name of output spike file
    std::string potentialLogName = argv[3]; // name of output potential file
    int gridWidth = std::atoi(argv[4]); // pixel grid width
    int gridHeight = std::atoi(argv[5]); // pixel grid height
    int numberOfNeurons = std::atoi(argv[6]); // number of neurons inside the reservoir
    float weightMean = std::atof(argv[7]); // gaussian mean for weights
    float weightStdDev = std::atof(argv[8]); // gaussian standard deviation for weights
    int feedforwardProbability = std::atoi(argv[9]); // percentage likelihood of feedforward connections
    int feedbackProbability = std::atoi(argv[10]); // percentage likelihood of feedback connections
    int selfExcitationProbability = std::atoi(argv[11]); // percentage likelihood of self-excitation
    float resetCurrent = std::atof(argv[12]); // current step function reset value (integration time)
    float decayPotential = std::atof(argv[13]); // time constant for membrane potential (decay)
    int refractoryPeriod = std::atoi(argv[14]); // neuron inactive for specified time after each spike
    bool wta = std::atoi(argv[15]); // winner-takes-all algorithm
    bool homeostasis = std::atoi(argv[16]); // threshold adaptation to firing rate

    // ----- IMPORTING DATA -----
    hummus::DataParser parser;
    auto data = parser.readData(dataPath);

    //  ----- INITIALISING THE NETWORK -----
    hummus::SpikeLogger spikeLog(spikeLogName);
    hummus::PotentialLogger potentialLog(potentialLogName);
    hummus::Network network({&spikeLog, &potentialLog});

    //  ----- CREATING THE NETWORK -----

    // pixel grid layer
    network.add2dLayer<hummus::InputNeuron>(gridWidth, gridHeight, 1, {});

    // reservoir layer
    network.addReservoir<hummus::LIF>(numberOfNeurons, weightMean, weightStdDev, feedforwardProbability, feedbackProbability, selfExcitationProbability, false, homeostasis, resetCurrent, decayPotential, refractoryPeriod, wta);

    // initialising the potentialLoggers
    potentialLog.neuronSelection(network.getLayers()[1]);

    //  ----- RUNNING THE NETWORK ASYNCHRONOUSLY-----
    network.run(&data, 0);

    //  ----- EXITING APPLICATION -----
    return 0;
}
