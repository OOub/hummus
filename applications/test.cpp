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

#include "../source/randomDistributions/normal.hpp"
#include "../source/randomDistributions/cauchy.hpp"
#include "../source/randomDistributions/lognormal.hpp"
#include "../source/randomDistributions/uniform.hpp"

#include "../source/GUI/qtDisplay.hpp"

#include "../source/neurons/input.hpp"
#include "../source/neurons/decisionMaking.hpp"
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

#include "../source/synapticKernels/exponential.hpp"

int main(int argc, char** argv) {

    //  ----- INITIALISING THE NETWORK -----
    hummus::QtDisplay qtDisplay;
    hummus::SpikeLogger spikeLog("testSpikeLog.bin");
    hummus::Network network({&spikeLog}, &qtDisplay);

    //  ----- CREATING THE NETWORK -----
    auto exponential = network.makeSynapticKernel<hummus::Exponential>();
	
    // creating layers of neurons
    network.addLayer<hummus::Input>(1, {}, nullptr);
    network.addLayer<hummus::LIF>(1, {}, &exponential, false, 20, 3, true);
    
    //  ----- CONNECTING THE NETWORK -----
    network.allToAll(network.getLayers()[0], network.getLayers()[1], hummus::Normal(1./2, 0));
	
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
	
	//  ----- SAVING THE NETWORK -----
    network.save("testSave");
	
    //  ----- EXITING APPLICATION -----
    return 0;
}
