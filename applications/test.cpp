/*
 * test.cpp
 * Adonis - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: Example of a basic spiking neural network.
 */

#include <iostream>

#include "../source/core.hpp"

#include "../source/GUI/qtDisplay.hpp"

#include "../source/neurons/inputNeuron.hpp"
#include "../source/neurons/decisionMakingNeuron.hpp"
#include "../source/neurons/leakyIntegrateAndFire.hpp"

#include "../source/addOns/spikeLogger.hpp"
#include "../source/addOns/predictionLogger.hpp"
#include "../source/addOns/myelinPlasticityLogger.hpp"
#include "../source/addOns/analysis.hpp"

#include "../source/learningRules/myelinPlasticity.hpp"
#include "../source/learningRules/rewardModulatedSTDP.hpp"
#include "../source/learningRules/stdp.hpp"

int main(int argc, char** argv) {

    //  ----- INITIALISING THE NETWORK -----
    adonis::QtDisplay qtDisplay;
    adonis::SpikeLogger spikeLog("testNetworkLog.bin");
    adonis::Network network({&spikeLog}, &qtDisplay);

    //  ----- CREATING THE NETWORK -----
    
    // creating layers of neurons
    network.addLayer<adonis::InputNeuron>(1, 1, 1, {});
    network.addLayer<adonis::LIF>(1, 1, 1, {}, false, 10, 5);

    //  ----- CONNECTING THE NETWORK -----
    network.allToAll(network.getLayers()[0], network.getLayers()[1], 1, 0, 10, 2);
    network.lateralInhibition(network.getLayers()[1], -1);

    //  ----- INJECTING SPIKES -----
    network.injectSpike(0, 10);
    network.injectSpike(0, 30);

    //  ----- DISPLAY SETTINGS -----
    qtDisplay.useHardwareAcceleration(true);
    qtDisplay.setTimeWindow(100);
    qtDisplay.trackNeuron(1);

    //  ----- RUNNING THE NETWORK -----
    network.run(100, 0.1);

    //  ----- EXITING APPLICATION -----
    return 0;
}
