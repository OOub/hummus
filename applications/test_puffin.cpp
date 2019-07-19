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
    hummus::Network network;
    network.makeGUI<hummus::PuffinDisplay>();
    auto input = network.makeLayer<hummus::Parrot>(1, {});

    int repetitions = 3600;
    int time_between_spikes = 5;
    int runtime = repetitions*time_between_spikes;

    for (auto i=0; i<repetitions; i++) {
        network.injectSpike(0, 1+time_between_spikes*i);
    }


    //  ----- RUNNING THE NETWORK -----
    network.verbosity(2);
    network.run(runtime, 0.1);

    return 0;
}
