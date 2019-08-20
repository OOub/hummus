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
#include "../source/GUI/display.hpp"
#include "../source/neurons/parrot.hpp"
#include "../source/neurons/decisionMaking.hpp"
#include "../source/neurons/LIF.hpp"
#include "../source/learningRules/myelinPlasticity.hpp"

int main(int argc, char** argv) {
    hummus::Network network;
    network.makeAddon<hummus::MyelinPlasticityLogger>("rates_mpLog.bin");

    auto& display = network.makeGUI<hummus::Display>();
    auto& mp = network.makeAddon<hummus::MyelinPlasticity>();

    auto input = network.makeLayer<hummus::LIF>(4, {}, 0, 200, 10, false, false);
    auto output = network.makeLayer<hummus::LIF>(1, {&mp}, 3, 200, 10, false, false);

    network.allToAll<hummus::Exponential>(input, output, 1, hummus::Normal(1./3, 0, 5, 3), 100);
    network.lateralInhibition<hummus::Exponential>(output, 1, hummus::Normal(-1, 0, 0, 1), 100);

    int repetitions = 500;
    int time_between_spikes = 100;
    int runtime = repetitions*time_between_spikes+10;

    for (auto i=0; i<repetitions; i++) {
        network.injectSpike(0, 10+time_between_spikes*i);
        network.injectSpike(1, 15+time_between_spikes*i);
        network.injectSpike(2, 20+time_between_spikes*i);
    }

    display.setTimeWindow(1100);
    display.trackNeuron(4);
    display.plotCurrents(true);

    network.verbosity(2);
    network.run(runtime, 0.1);

    return 0;
}
