/*
 * test.cpp
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
#include "../source/rand.hpp"
#include "../source/GUI/qtDisplay.hpp"

#include "../source/neurons/inputNeuron.hpp"
#include "../source/neurons/decisionMakingNeuron.hpp"
#include "../source/neurons/LIF.hpp"
#include "../source/neurons/IF.hpp"

#include "../source/addOns/spikeLogger.hpp"
#include "../source/addOns/potentialLogger.hpp"
#include "../source/addOns/classificationLogger.hpp"
#include "../source/addOns/myelinPlasticityLogger.hpp"
#include "../source/addOns/analysis.hpp"

#include "../source/learningRules/myelinPlasticity.hpp"
#include "../source/learningRules/rewardModulatedSTDP.hpp"
#include "../source/learningRules/timeInvariantSTDP.hpp"
#include "../source/learningRules/stdp.hpp"

int main(int argc, char** argv) {

    //  ----- INITIALISING THE NETWORK -----
    hummus::QtDisplay qtDisplay;
    hummus::SpikeLogger spikeLog("testSpikeLog.bin");
    hummus::Network network({&spikeLog}, &qtDisplay);

    //  ----- CREATING THE NETWORK -----
    
    // creating layers of neurons
    
    network.addLayer<hummus::InputNeuron>(1, {});
    network.addLayer<hummus::LIF>(2, {}, true, false, 10, 20, 0, false);
    
    //  ----- CONNECTING THE NETWORK -----
    network.allToAll(network.getLayers()[0], network.getLayers()[1], hummus::Rand(1./2, 0.1));
    network.lateralInhibition(network.getLayers()[1], -1);
	
    //  ----- INJECTING SPIKES -----
    network.injectSpike(0, 10);
    network.injectSpike(0, 11);
    network.injectSpike(0, 30);
    
    //  ----- DISPLAY SETTINGS -----
    qtDisplay.useHardwareAcceleration(true);
    qtDisplay.setTimeWindow(100);
    qtDisplay.trackNeuron(1);
	
    //  ----- RUNNING THE NETWORK -----
    network.run(100, 0.1);
    
    network.save("testSave");
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
