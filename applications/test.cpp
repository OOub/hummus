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

#include "../source/neurons/parrot.hpp"
#include "../source/neurons/decisionMaking.hpp"
#include "../source/neurons/LIF.hpp"

#include "../source/addons/spikeLogger.hpp"
#include "../source/addons/potentialLogger.hpp"
#include "../source/addons/classificationLogger.hpp"
#include "../source/addons/weightMaps.hpp"
#include "../source/addons/analysis.hpp"

#include "../source/learningRules/myelinPlasticity.hpp"
#include "../source/learningRules/rewardModulatedSTDP.hpp"
#include "../source/learningRules/timeInvariantSTDP.hpp"
#include "../source/learningRules/stdp.hpp"

#include "../source/synapses/dirac.hpp"
#include "../source/synapses/pulse.hpp"
#include "../source/synapses/exponential.hpp"

int main(int argc, char** argv) {

    //  ----- INITIALISING THE NETWORK -----
    hummus::Network network;
    
    //  ----- INITIALISING ADD-ONS -----
    network.makeAddon<hummus::SpikeLogger>("spikeLog.bin");
    
    // ----- INITIALISING GUI -----
    auto& display = network.makeGUI<hummus::QtDisplay>();
    
    //  ----- CREATING THE NETWORK -----
    // creating layers of neurons
    auto input = network.makeLayer<hummus::Parrot>(1, {});
    auto output = network.makeLayer<hummus::LIF>(2, {}, false, 200, 10, 3, false);

    //  ----- CONNECTING THE NETWORK -----
    network.allToAll<hummus::Exponential>(input, output, 1, hummus::Normal(1./2, 0, 1, 0.5), 100, hummus::synapseType::excitatory);
    network.lateralInhibition<hummus::Exponential>(output, 1, hummus::Normal(-1, 0), 100);
	
    //  ----- INJECTING SPIKES -----
    network.injectSpike(0, 10);
    network.injectSpike(0, 12);
    network.injectSpike(0, 30);

    //  ----- DISPLAY SETTINGS -----
    display.setTimeWindow(100);
    display.trackNeuron(1);

    network.turnOffLearning(2);
    
    //  ----- RUNNING THE NETWORK -----
    network.verbosity(2);
    network.run(100, 0.1);

    //  ----- SAVE THE NETWORK IN A JSON FILE -----
    network.save("testSave");
    
    //  ----- EXITING APPLICATION -----
    return 0;
}
