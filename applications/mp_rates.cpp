/*
 * mp_rates.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: figuring out how to work with rates in the context of the myelin plasticity rule
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/dataParser.hpp"

#include "../source/randomDistributions/normal.hpp"

#include "../source/GUI/qt/qtDisplay.hpp"

#include "../source/neurons/parrot.hpp"
#include "../source/neurons/decisionMaking.hpp"
#include "../source/neurons/LIF.hpp"

#include "../source/addons/spikeLogger.hpp"
#include "../source/addons/potentialLogger.hpp"
#include "../source/addons/classificationLogger.hpp"
#include "../source/addons/weightMaps.hpp"
#include "../source/addons/analysis.hpp"

#include "../source/learningRules/myelinPlasticity.hpp"

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
    hummus::DataParser parser;
    
    // creating layers of neurons
    auto input = network.makeLayer<hummus::Parrot>(3, {});
    auto output = network.makeLayer<hummus::LIF>(1, {}, false, 200, 10, 1, false);

    //  ----- CONNECTING THE NETWORK -----
    network.allToAll<hummus::Exponential>(input, output, 1, hummus::Normal(1., 0, 1, 0), 100);
	
    //  ----- INJECTING SPIKES -----
    int repetitions = 10;
    int time_between_spikes = 100;
    int runtime = repetitions*time_between_spikes+100;
    
    for (auto i=0; i<repetitions; i++) {
        network.injectPoissonSpikes(0, 10+time_between_spikes*i, 1, 0.1, 0.5);
        network.injectPoissonSpikes(1, 15+time_between_spikes*i, 1, 0.1, 0.5);
        network.injectPoissonSpikes(2, 20+time_between_spikes*i, 1, 0.1, 0.5);
    }
    
    //  ----- DISPLAY SETTINGS -----
    display.setTimeWindow(500);
    display.trackNeuron(3);

    //  ----- RUNNING THE NETWORK -----
    network.verbosity(1);
    network.run(runtime, 0.1);
    
    //  ----- EXITING APPLICATION -----
    return 0;
}
