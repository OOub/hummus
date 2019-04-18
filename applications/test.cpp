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
#include "../source/dataParser.hpp"

#include "../source/randomDistributions/normal.hpp"
#include "../source/randomDistributions/cauchy.hpp"
#include "../source/randomDistributions/lognormal.hpp"
#include "../source/randomDistributions/uniform.hpp"

#include "../source/GUI/qt/qtDisplay.hpp"
#include "../source/GUI/puffin/puffinDisplay.hpp"

#include "../source/neurons/input.hpp"
#include "../source/neurons/decisionMaking.hpp"
#include "../source/neurons/LIF.hpp"
#include "../source/neurons/IF.hpp"

#include "../source/addons/spikeLogger.hpp"
#include "../source/addons/potentialLogger.hpp"
#include "../source/addons/classificationLogger.hpp"
#include "../source/addons/myelinPlasticityLogger.hpp"
#include "../source/addons/analysis.hpp"

#include "../source/learningRules/myelinPlasticity.hpp"
#include "../source/learningRules/rewardModulatedSTDP.hpp"
#include "../source/learningRules/timeInvariantSTDP.hpp"
#include "../source/learningRules/stdp.hpp"

#include "../source/synapticKernels/exponential.hpp"

int main(int argc, char** argv) {

    //  ----- INITIALISING THE NETWORK -----
    hummus::Network network;
    
    //  ----- INITIALISING ADD-ONS -----
    network.makeAddon<hummus::SpikeLogger>("spikeLog.bin");
    
    // ----- INITIALISING GUI -----
    auto& display = network.makeGUI<hummus::QtDisplay>();
    
    //  ----- CREATING THE NETWORK -----
    hummus::DataParser parser;

    // creating a synaptic kernel
    auto& exponential = network.makeSynapticKernel<hummus::Exponential>();

    // creating layers of neurons
    network.makeLayer<hummus::Input>(1, {});
    network.makeLayer<hummus::LIF>(2, {}, &exponential, false, 20, 3, true);

    //  ----- CONNECTING THE NETWORK -----
    network.allToAll(network.getLayers()[0], network.getLayers()[1], hummus::Normal(1./2, 0));
	
    //  ----- INJECTING SPIKES -----
    network.injectSpike(0, 10);
    network.injectSpike(0, 11);
    network.injectSpike(0, 30);

    //  ----- DISPLAY SETTINGS -----
    display.setTimeWindow(100);
    display.trackNeuron(1);

    //  ----- RUNNING THE NETWORK -----
    network.verbosity(1);
    network.run(100, 0.1);

	//  ----- SAVING THE NETWORK -----
    network.save("testSave");

    //  ----- EXITING APPLICATION -----
    return 0;
}
