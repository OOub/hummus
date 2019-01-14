/*
 * testNetwork.cpp
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

int main(int argc, char** argv)
{

    //  ----- INITIALISING THE NETWORK -----
    adonis::QtDisplay qtDisplay;
    adonis::Network network(&qtDisplay);

    //  ----- NETWORK PARAMETERS -----
    float runtime = 100;
    float timestep = 0.1;

    //  ----- CREATING THE NETWORK -----
    
    // creating layers of neurons
    network.addLayer<adonis::InputNeuron>(1, 1, 1, {});
    network.addLayer<adonis::LIF>(1, 1, 1, {});

    //  ----- CONNECTING THE NETWORK -----
    network.allToAll(network.getLayers()[0], network.getLayers()[1], 1, 0, 10, 2);
    network.lateralInhibition(network.getLayers()[1], -1);

    //  ----- INJECTING SPIKES -----
    network.injectSpike(network.getNeurons()[0]->prepareInitialSpike(10));
    network.injectSpike(network.getNeurons()[0]->prepareInitialSpike(30));

    //  ----- DISPLAY SETTINGS -----
    qtDisplay.useHardwareAcceleration(true);
    qtDisplay.setTimeWindow(runtime);
    qtDisplay.trackNeuron(1);

    //  ----- RUNNING THE NETWORK -----
    network.run(runtime, timestep);

    //  ----- EXITING APPLICATION -----
    return 0;
}
