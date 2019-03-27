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
    hummus::QtDisplay display;
    hummus::SpikeLogger spikeLog("spikeLog.bin");
    hummus::ClassificationLogger classificationLog("classificationLog.bin");
    hummus::PotentialLogger potentialLog("potentialLog.bin");

    hummus::Network network({&spikeLog, &classificationLog, &potentialLog}, &display);

    //  ----- CREATING THE NETWORK -----
    hummus::DataParser parser;

    auto exponential = network.makeSynapticKernel<hummus::Exponential>();

    // creating layers of neurons
    network.addLayer<hummus::Input>(1, {}, nullptr);
    network.addLayer<hummus::LIF>(1, {}, &exponential, false, 20, 3, true);

    //  ----- CONNECTING THE NETWORK -----
    network.allToAll(network.getLayers()[0], network.getLayers()[1], hummus::Normal(0, 0));

    //  ----- INJECTING SPIKES -----
    network.injectSpike(0, 10);
    network.injectSpike(0, 11);
    network.injectSpike(0, 30);

    //  ----- DISPLAY SETTINGS -----
    qtDisplay.useHardwareAcceleration(true);
    qtDisplay.setTimeWindow(100);
    qtDisplay.trackNeuron(1);

    //  ----- RUNNING THE NETWORK -----
    network.turnOffLearning(0);
    potentialLog.neuronSelection(1);
    network.run(100, 0.1);

	//  ----- SAVING THE NETWORK -----
    network.save("testSave");

    //  ----- EXITING APPLICATION -----
    return 0;
}
