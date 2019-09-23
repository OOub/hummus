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
#include "../source/neurons/cuba_lif.hpp"
#include "../source/learningRules/myelinPlasticity.hpp"

int main(int argc, char** argv) {
    hummus::Network network;
    network.make_addon<hummus::MyelinPlasticityLogger>("rates_mpLog.bin");

    auto& display = network.make_gui<hummus::Display>();
    auto& mp = network.make_addon<hummus::MyelinPlasticity>();

    auto input = network.make_layer<hummus::CUBA_LIF>(4, {}, 0, 200, 10, false, false);
    auto output = network.make_layer<hummus::CUBA_LIF>(1, {&mp}, 3, 200, 10, false, false);

    network.all_to_all<hummus::Exponential>(input, output, 1, hummus::Normal(1./3, 0, 5, 3), 100);
    network.lateral_inhibition<hummus::Exponential>(output, 1, hummus::Normal(-1, 0, 0, 1), 100);

    int repetitions = 500;
    int time_between_spikes = 100;
    int runtime = repetitions*time_between_spikes+10;

    for (auto i=0; i<repetitions; i++) {
        network.inject_spike(0, 10+time_between_spikes*i);
//        network.inject_spike(0, 12+time_between_spikes*i);
        network.inject_spike(1, 15+time_between_spikes*i);
        network.inject_spike(2, 20+time_between_spikes*i);
    }

    display.set_time_window(1100);
    display.track_neuron(4);
    display.plot_currents(true);

    network.verbosity(2);
    network.run(runtime, 0.1f);

    return 0;
}
