/*
 * rc.cpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 26/02/2019
 *
 * Information: Reservoir computer (without the readout function)
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/dataParser.hpp

#include "../source/neurons/inputNeuron.hpp"
#include "../source/neurons/LIF.hpp"
#include "../source/neurons/IF.hpp"

#include "../source/addOns/spikeLogger.hpp"
#include "../source/addOns/potentialLogger.hpp"

int main(int argc, char** argv) {
    
    // ----- RESERVOIR PARAMETERS -----
    int numberOfNeurons = 10;
    float weightMean = 1;
    float weightStd = 1;
    int feedforwardProbability = 100;
    int feedbackProbability = 100;
    int selfExcitationProbability = 100;
    
    // ----- IF PARAMETERS -----
    bool homeostasis = true; // changes threshold according to neuorn firing rate
    bool timeDependentCurrent = true;
    float resetCurrent = 10;
    int refractoryPeriod = 3;
    bool wta = false; // winner-takes-all algorithm
    
    // ----- IMPORTING DATA -----
    adonis::DataParser parser;
    auto data = parser.readData("path to file");
    
    //  ----- INITIALISING THE NETWORK -----
    adonis::QtDisplay qtDisplay;
    adonis::SpikeLogger spikeLog("rcSpike.bin");
    adonis::PotentialLogger reservoirPLog("rervoirPotential.bin");
    adonis::Network network({&spikeLog, &reservoirPLog});

    //  ----- CREATING THE NETWORK -----
    
    // creating layers of neurons
//    network.add2DLayer<adonis::InputNeuron>(1, 1, 1, {});
    
    network.addReservoir<adonis::IF>(numberOfNeurons, weightMean, weightStd, feedforwardProbability, feedbackProbability, selfExcitationProbability, timeDependentCurrent, homeostasis, resetCurrent, refractoryPeriod, wta);
    
    // initialising the potentialLoggers
    reservoirPLog.neuronSelection(network.getLayers()[1]);
	
    //  ----- RUNNING THE NETWORK -----
    network.run(&data, 0.1);
    
    //  ----- EXITING APPLICATION -----
    return 0;
}
