/*
 * mp_1D.cpp
 * Hummus - spiking neural network simulator
 *
 * Created by Omar Oubari.
 * Email: omar.oubari@inserm.fr
 * Last Version: 14/01/2019
 *
 * Information: Example of a spiking neural network that can learn one dimensional patterns.
 */

#include <iostream>

#include "../source/core.hpp"
#include "../source/GUI/display.hpp"
#include "../source/addons/spike_logger.hpp"
#include "../source/addons/potential_logger.hpp"
#include "../source/learning_rules/myelin_plasticity_v1.hpp"
#include "../source/neurons/parrot.hpp"
#include "../source/neurons/cuba_lif.hpp"

int main(int argc, char** argv) {
    
    //  ----- READING TRAINING DATA FROM FILE -----
	hummus::DataParser dataParser;

    auto dataset = dataParser.load_data("/Users/omaroubari/Datasets/1D_patterns/oneD_10neurons_4patterns.npy", "");

    //  ----- INITIALISING THE NETWORK -----
    hummus::Network network;

    auto& display = network.make_gui<hummus::Display>();
    network.make_addon<hummus::SpikeLogger>("1D_spikeLog.bin");
    network.make_addon<hummus::MyelinPlasticityLogger>("1D_mpLog.bin");
    auto& vlog = network.make_addon<hummus::PotentialLogger>("1D_vLog.bin");

    //  ----- NETWORK PARAMETERS -----
	float conductance        = 250;
    float leakageConductance = 10;
    int inputNeurons         = 10;
    int layer1Neurons        = 4;
	bool burst               = false;
	bool homeostasis         = false;
    bool wta                 = true;
    
	//  ----- INITIALISING THE LEARNING RULE -----
    auto& mp = network.make_addon<hummus::MP_1>(100, 2);

    //  ----- CREATING THE NETWORK -----
    auto input = network.make_layer<hummus::Parrot>(inputNeurons, {}, 0 , 100);
    auto output = network.make_layer<hummus::CUBA_LIF>(layer1Neurons, {&mp}, 3, conductance, leakageConductance, wta, homeostasis, burst);

	//  ----- CONNECTING THE NETWORK -----
    network.all_to_all<hummus::Exponential>(input, output, 1, hummus::Normal(0.1, 0, 10, 3), 100, 10, 100);

    //  ----- DISPLAY SETTINGS -----
    display.set_time_window(5000);
    display.track_neuron(12);
    display.plot_currents();
    
    network.turn_off_learning(80000);
    network.verbosity(0);
    vlog.activate_for({10,11,12,13});
    
    //  ----- RUNNING THE NETWORK -----
    network.run_data(dataset.spikes, 0.1);

    //  ----- EXITING APPLICATION -----
    return 0;
}
